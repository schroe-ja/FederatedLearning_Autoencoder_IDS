#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"

// TFLite Includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Models (Stelle sicher, dass diese Header nun deine INT8-Modelle enthalten!)
#include "model.h"
#include "model_encoder.h" 

// Test Data
#include "mcu_test_data.h" 
// Data for Calibration of treshold
#include "mcu_val_data.h" 
//Values for DT
#include "tree_model.h"

static const char *TAG = "ESP_INFERENCE_INT8";

#define N_FEATURES 67
const int kArenaSize = 30 * 1024; // Bei INT8 brauchst du eventuell sogar etwas mehr wegen Scratch-Buffern!
alignas(16) uint8_t tensor_arena_ae[kArenaSize];
alignas(16) uint8_t tensor_arena_enc[kArenaSize];

// Normalization Constants for the Decision Tree features
float mu_val = 0.0f;
float sigma_val = 0.0f;
float threshold = 0.0f;

struct SampleScore {
    float score;
    int label; 
};

bool compareScores(const SampleScore& a, const SampleScore& b) { return a.score > b.score; }

double safe_div(double numerator, double denominator) { return (denominator == 0.0) ? 0.0 : numerator / denominator; }
double calculate_FNR(int tp, int fn) { return safe_div((double)fn, (double)(tp + fn)); }
double calculate_FPR(int fp, int tn) { return safe_div((double)fp, (double)(fp + tn)); }
double calculate_FDR(int fp, int tp) { return safe_div((double)fp, (double)(fp + tp)); }

double calculate_MCC(int tp, int tn, int fp, int fn) {
    double numerator = ((double)tp * tn) - ((double)fp * fn);
    double denominator_sq = ((double)tp + fp) * ((double)tp + fn) * ((double)tn + fp) * ((double)tn + fn);
    if (denominator_sq == 0.0) return 0.0;
    return numerator / sqrt(denominator_sq);
}

double entropy_term(double p) { return (p <= 0.0 || p >= 1.0) ? 0.0 : -p * log2(p); }

double calculate_CID(int tp, int tn, int fp, int fn) {
    double total = (double)(tp + tn + fp + fn);
    if (total == 0.0) return 0.0;
    double p_intrusion = (tp + fn) / total;
    double p_normal = (tn + fp) / total;
    double p_alarm = (tp + fp) / total;
    double p_no_alarm = (tn + fn) / total;
    double H_X = entropy_term(p_intrusion) + entropy_term(p_normal);
    if (H_X == 0.0) return 0.0;
    double H_X_given_Alarm = entropy_term(safe_div(tp, tp + fp)) + entropy_term(safe_div(fp, tp + fp));
    double H_X_given_NoAlarm = entropy_term(safe_div(fn, tn + fn)) + entropy_term(safe_div(tn, tn + fn));
    double H_X_given_Y = (p_alarm * H_X_given_Alarm) + (p_no_alarm * H_X_given_NoAlarm);
    return (H_X - H_X_given_Y) / H_X;
}
double calculate_Accuracy(int tp, int tn, int fp, int fn) {
    double total = (double)(tp + tn + fp + fn);
    return safe_div((double)(tp + tn), total);
}

double calculate_F1(int tp, int fp, int fn) {
    double precision = safe_div((double)tp, (double)(tp + fp));
    double recall = safe_div((double)tp, (double)(tp + fn));
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculate_roc_auc_and_ap(std::vector<SampleScore>& scores, float& out_auc, float& out_ap) {
    if (scores.empty()) { out_auc = 0.0f; out_ap = 0.0f; return; }
    std::sort(scores.begin(), scores.end(), compareScores);
    int num_pos = 0, num_neg = 0;
    for (const auto& s : scores) { if (s.label == 1) num_pos++; else num_neg++; }
    if (num_pos == 0 || num_neg == 0) { out_auc = 0.0f; out_ap = 0.0f; return; }
    
    float tp = 0.0f, fp = 0.0f, prev_tp = 0.0f, prev_fp = 0.0f, prev_recall = 0.0f, auc = 0.0f, ap = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i].label == 1) tp++; else fp++;
        float precision = (tp + fp > 0) ? (tp / (tp + fp)) : 0.0f;
        float recall = tp / num_pos;
        if (i == scores.size() - 1 || scores[i].score != scores[i + 1].score) {
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0f;
            prev_tp = tp; prev_fp = fp;
        }
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    out_auc = auc / (num_pos * num_neg);
    out_ap = ap;
}

// =========================================================
// NEU: Quantisierungs-Hilfsfunktionen für INT8
// =========================================================
int8_t quantize_float_to_int8(float x, float scale, int zero_point) {
    int32_t q = (int32_t)roundf(x / scale) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

float dequantize_int8_to_float(int8_t q, float scale, int zero_point) {
    return ((float)q - zero_point) * scale;
}
// =========================================================

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "--- Starting FULL INT8 Anomaly Detection Engine ---");

    const tflite::Model* model_ae = tflite::GetModel(model_tflite);
    const tflite::Model* model_enc = tflite::GetModel(model_enc_tflite);

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddLeakyRelu();
    resolver.AddReshape(); 
    // Für INT8 brauchst du explizit diese Layer:
    resolver.AddQuantize(); 
    resolver.AddDequantize(); 

    static tflite::MicroInterpreter interpreter_ae(model_ae, resolver, tensor_arena_ae, kArenaSize);
    static tflite::MicroInterpreter interpreter_enc(model_enc, resolver, tensor_arena_enc, kArenaSize); // Delete for no DT

    if (interpreter_ae.AllocateTensors() != kTfLiteOk || interpreter_enc.AllocateTensors() != kTfLiteOk) { // Delete interpreter_enc for no DT
        ESP_LOGE(TAG, "Failed to allocate tensors. Consider increasing kArenaSize for INT8 scratch buffers!");
        return;
    }

    TfLiteTensor* input_ae = interpreter_ae.input(0);
    ESP_LOGI(TAG, "Model Input Type: %d (1=Float32, 9=INT8)", input_ae->type);

    if (input_ae->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Achtung: Das Modell erwartet kein INT8 als Input! Bitte Python Export prüfen.");
    }

    // Hole die Quantisierungs-Parameter des Modells
    float input_scale_ae = input_ae->params.scale;
    int input_zp_ae = input_ae->params.zero_point;
    
    TfLiteTensor* output_ae_tensor = interpreter_ae.output(0);
    float output_scale_ae = output_ae_tensor->params.scale;
    int output_zp_ae = output_ae_tensor->params.zero_point;

    TfLiteTensor* input_enc_tensor = interpreter_enc.input(0); // Delete for no DT
    float input_scale_enc = input_enc_tensor->params.scale; // Delete for no DT
    int input_zp_enc = input_enc_tensor->params.zero_point; // Delete for no DT

    TfLiteTensor* output_enc_tensor = interpreter_enc.output(0); // Delete for no DT
    float output_scale_enc = output_enc_tensor->params.scale; // Delete for no DT
    int output_zp_enc = output_enc_tensor->params.zero_point; // Delete for no DT

    // ==========================================
    // PHASE 1: HARDWARE-KALIBRIERUNG (INT8)
    // ==========================================
    ESP_LOGI(TAG, "Starte INT8 Kalibrierung...");

    std::vector<float> calib_mses;
    calib_mses.reserve(MCU_val_SAMPLES);
    
    for (int i = 0; i < MCU_val_SAMPLES; ++i) {
        if (i % 50 == 0) vTaskDelay(pdMS_TO_TICKS(10));    

        // 1. Manuelles Quantisieren der Float-Testdaten zu INT8
        for(int j=0; j<N_FEATURES; j++) {
            input_ae->data.int8[j] = quantize_float_to_int8(mcu_val_x[i][j], input_scale_ae, input_zp_ae);
        }

        if (interpreter_ae.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke AE failed during calibration!");
            break;
        }

        // 2. Manuelles Dequantisieren der INT8-Ausgabe zurück zu Float für mse
        float mse = 0.0f;
        for(int j=0; j<N_FEATURES; j++) {
            float recon_f = dequantize_int8_to_float(output_ae_tensor->data.int8[j], output_scale_ae, output_zp_ae);
            float diff = mcu_val_x[i][j] - recon_f;
            mse += (diff * diff);
        }
        mse /= (float)N_FEATURES;
        calib_mses.push_back(mse);
    }

    float sum_mse = 0.0f;
    for (float m : calib_mses) sum_mse += m;
    mu_val = sum_mse / calib_mses.size();

    float sq_sum_mse = 0.0f;
    for (float m : calib_mses) sq_sum_mse += powf(m - mu_val, 2);
    sigma_val = sqrtf(sq_sum_mse / calib_mses.size());

    std::sort(calib_mses.begin(), calib_mses.end());
    float percentile = 0.85f; 
    float position = percentile * (calib_mses.size() - 1);
    int lower_idx = (int)floor(position);
    int upper_idx = (int)ceil(position);
    float fraction = position - lower_idx;

    if (lower_idx == upper_idx) {
        threshold = calib_mses[lower_idx];
    } else {
        threshold = calib_mses[lower_idx] + fraction * (calib_mses[upper_idx] - calib_mses[lower_idx]);
    }

    ESP_LOGI(TAG, "Kalibrierung beendet!");
    ESP_LOGI(TAG, "INT8 mu: %.6f", mu_val);
    ESP_LOGI(TAG, "INT8 sigma: %.6f", sigma_val);
    ESP_LOGI(TAG, "INT8 threshold: %.6f", threshold);

    // ==========================================
    // PHASE 2: INFERENZ (INT8)
    // ==========================================
    while (true) {
        std::vector<SampleScore> ae_scores, dt_scores;
        int tp_ae = 0, fp_ae = 0, tn_ae = 0, fn_ae = 0;
        int tp_dt = 0, fp_dt = 0, tn_dt = 0, fn_dt = 0;
        int64_t total_start = esp_timer_get_time();
        int64_t cumulative_inference_time = 0;

        for (int i = 0; i < MCU_TEST_SAMPLES; ++i) {
            if (i % 50 == 0) vTaskDelay(pdMS_TO_TICKS(10)); 
            
            int64_t sample_start = esp_timer_get_time();
            const float* raw_input = mcu_test_x[i];
            int y_true = mcu_test_y[i];

            // --- 1. Run Autoencoder (INT8) ---
            for(int j=0; j<N_FEATURES; j++) {
                input_ae->data.int8[j] = quantize_float_to_int8(raw_input[j], input_scale_ae, input_zp_ae);
            }
            interpreter_ae.Invoke();

            float mse = 0.0f;
            for(int j=0; j<N_FEATURES; j++) {
                float recon_f = dequantize_int8_to_float(output_ae_tensor->data.int8[j], output_scale_ae, output_zp_ae);
                float diff = raw_input[j] - recon_f;
                mse += diff * diff;
            }
            mse /= (float)N_FEATURES;

            // --- 2. Run Encoder (INT8) --- // Delete for no DT
            for(int j=0; j<N_FEATURES; j++) {
                input_enc_tensor->data.int8[j] = quantize_float_to_int8(raw_input[j], input_scale_enc, input_zp_enc);
            }
            interpreter_enc.Invoke();

            //float encoded[6]; 
            //for(int j=0; j<6; j++) {
            //    encoded[j] = dequantize_int8_to_float(output_enc_tensor->data.int8[j], output_scale_enc, output_zp_enc);
            //}

            // --- 3. Decision Tree Features ---
            float error_norm = mse; // Delete for no DT
            
            float sum = 0, sum_sq_diff = 0; 
            for(int j=0; j<N_FEATURES; j++) sum += raw_input[j];
            float mean = sum / N_FEATURES;
            for(int j=0; j<N_FEATURES; j++) {
                float d = raw_input[j] - mean;
                sum_sq_diff += d*d;
            }
            float stats = sqrtf(sum_sq_diff / N_FEATURES); // Delete for no DT

            int16_t features[8]; // Delete for no DT
            for(int j=0; j<6; j++) {
                float enc_f = dequantize_int8_to_float(output_enc_tensor->data.int8[j], output_scale_enc, output_zp_enc);   // // Delete for no DT
                features[j] = (int16_t)std::clamp(enc_f  * 10000.0f, -32768.0f, 32767.0f);
            }

            features[6] = (int16_t)std::clamp(error_norm * 10000.0f, -32768.0f, 32767.0f); // Delete for no DT
            features[7] = (int16_t)std::clamp(stats * 10000.0f, -32768.0f, 32767.0f); // Delete for no DT
            // --- 4. Get Predictions ---
            int32_t y_pred_dt = my_model_predict(features,8); // Delete for no DT
            int y_pred_ae = (mse > threshold) ? 1 : 0;
            // Wahrscheinlichkeiten für ROC/AP (da es nur 1 Baum ist, wird proba[1] entweder 0.0 oder 1.0 sein)
            float proba[2]; // Delete for no DT
            my_model_predict_proba(features, 8, proba, 2); // Delete for no DT
            float dt_score = proba[1]; // Score für Klasse 1 (Anomalie) // Delete for no DT
            int64_t sample_end = esp_timer_get_time();
            cumulative_inference_time += (sample_end - sample_start);

            ae_scores.push_back({mse, y_true});
            dt_scores.push_back({dt_score, y_true}); // Delete for no DT

            if (y_true == 1 && y_pred_ae == 1) tp_ae++;
            else if (y_true == 0 && y_pred_ae == 1) fp_ae++;
            else if (y_true == 0 && y_pred_ae == 0) tn_ae++;
            else if (y_true == 1 && y_pred_ae == 0) fn_ae++;

            if (y_true == 1 && y_pred_dt == 1) tp_dt++; // Delete for no DT
            else if (y_true == 0 && y_pred_dt == 1) fp_dt++; // Delete for no DT
            else if (y_true == 0 && y_pred_dt == 0) tn_dt++; // Delete for no DT
            else if (y_true == 1 && y_pred_dt == 0) fn_dt++; // Delete for no DT
        }

        float roc_auc_ae, ap_ae, roc_auc_dt, ap_dt; // Delete dt for no DT
        calculate_roc_auc_and_ap(ae_scores, roc_auc_ae, ap_ae);
        calculate_roc_auc_and_ap(dt_scores, roc_auc_dt, ap_dt); // Delete for no DT
        


        // --- Print Inference Report ---
        ESP_LOGI(TAG, "========================================");
        ESP_LOGI(TAG, "           INFERENCE RESULTS            ");
        ESP_LOGI(TAG, "========================================");
        ESP_LOGI(TAG, "Total Evaluated:        %d", MCU_TEST_SAMPLES);
        
        ESP_LOGI(TAG, "\n--- AUTOENCODER (Thresholding) ---");
        ESP_LOGI(TAG, "TP: %d | TN: %d | FP: %d | FN: %d", tp_ae, tn_ae, fp_ae, fn_ae);
        ESP_LOGI(TAG, "ROC AUC: %.4f | Avg Precision: %.4f", roc_auc_ae, ap_ae);
        ESP_LOGI(TAG, "FNR: %.4f | FPR: %.4f | FDR: %.4f", calculate_FNR(tp_ae, fn_ae), calculate_FPR(fp_ae, tn_ae), calculate_FDR(fp_ae, tp_ae));
        ESP_LOGI(TAG, "MCC: %.4f | C_ID: %.4f", calculate_MCC(tp_ae, tn_ae, fp_ae, fn_ae), calculate_CID(tp_ae, tn_ae, fp_ae, fn_ae));
        ESP_LOGI(TAG, "Accuracy: %.4f | F1-Score: %.4f", calculate_Accuracy(tp_ae, tn_ae, fp_ae, fn_ae), calculate_F1(tp_ae, fp_ae, fn_ae));
        ESP_LOGI(TAG, "\n--- DECISION TREE CLASSIFIER ---"); // Delete for no DT
        ESP_LOGI(TAG, "TP: %d | TN: %d | FP: %d | FN: %d", tp_dt, tn_dt, fp_dt, fn_dt);
        ESP_LOGI(TAG, "ROC AUC: %.4f | Avg Precision: %.4f", roc_auc_dt, ap_dt);
        ESP_LOGI(TAG, "FNR: %.4f | FPR: %.4f | FDR: %.4f", calculate_FNR(tp_dt, fn_dt), calculate_FPR(fp_dt, tn_dt), calculate_FDR(fp_dt, tp_dt));
        ESP_LOGI(TAG, "MCC: %.4f | C_ID: %.4f", calculate_MCC(tp_dt, tn_dt, fp_dt, fn_dt), calculate_CID(tp_dt, tn_dt, fp_dt, fn_dt));
        ESP_LOGI(TAG, "Accuracy: %.4f | F1-Score: %.4f", calculate_Accuracy(tp_dt, tn_dt, fp_dt, fn_dt), calculate_F1(tp_dt, fp_dt, fn_dt));
        ESP_LOGI(TAG, "========================================");

        ESP_LOGI(TAG, "========== RESOURCE REPORT ==========");
        ESP_LOGI(TAG, "STORAGE (Flash):");
        // Hier gibst du die Summe beider C-Arrays an (AE + Encoder) // Delete Encoder for no DT
        ESP_LOGI(TAG, "  - Combined Model Weights: 18.11 KB + 8.67 KB = 26.78 KB"); 

        ESP_LOGI(TAG, "RUNTIME MEMORY (RAM):");
        // Wichtig: Zeige die Auslastung beider Modelle!
        ESP_LOGI(TAG, "  - AE Arena (Used/Total):  %d / %d bytes", 
                interpreter_ae.arena_used_bytes(), kArenaSize);
        ESP_LOGI(TAG, "  - Enc Arena (Used/Total): %d / %d bytes", 
                interpreter_enc.arena_used_bytes(), kArenaSize);

        ESP_LOGI(TAG, "SYSTEM STABILITY:");
        ESP_LOGI(TAG, "  - Current Free Heap:      %u bytes", esp_get_free_heap_size());
        // DAS ist der Gold-Wert für die Thesis:
        ESP_LOGI(TAG, "  - Minimum Free Heap Ever: %u bytes", esp_get_minimum_free_heap_size()); 
        ESP_LOGI(TAG, "=====================================");

        int64_t total_end = esp_timer_get_time();
        float avg_inference_ms = (float)cumulative_inference_time / (float)MCU_TEST_SAMPLES / 1000.0f;
        float total_inference_ms = (float)(total_end - total_start) / 1000.0f;
               // --- Print Timing Report ---
        ESP_LOGI(TAG, "========== TIMING REPORT ==========");
        ESP_LOGI(TAG, "Total Batch Time:    %.2f ms", total_inference_ms);
        ESP_LOGI(TAG, "Avg. Time per Sample: %.4f ms", avg_inference_ms);
        ESP_LOGI(TAG, "Samples per Second:   %.2f Hz", 1000.0f / avg_inference_ms);
        ESP_LOGI(TAG, "===================================");        
        vTaskDelay(pdMS_TO_TICKS(1000000));
    }
}