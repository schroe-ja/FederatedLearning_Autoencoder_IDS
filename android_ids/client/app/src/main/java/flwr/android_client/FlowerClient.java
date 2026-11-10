package flwr.android_client;

import android.content.Context;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;

public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;

    public FlowerClient(Context context) {
        this.tlModel = new TransferLearningModelWrapper(context);
        this.context = context;
    }

    public ByteBuffer[] getWeights() {
        return tlModel.getParameters();
    }

    public Pair<ByteBuffer[], Integer> fit(ByteBuffer[] weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();
        return Pair.create(getWeights(), tlModel.getSize_Training());
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    /**
     * IDS adaptation: reads assets/data/partition_<id>_train.txt and _test.txt.
     * Each line: label,f1,...,f11  (label 0 = benign, 1 = malicious; features already scaled
     * by scale_features() in dataset_load_tpot.py). Created by export_partitions.py.
     */
    public void loadData(int device_id) {
        try {
            Log.d("FLOWERCLIENT_LOAD", "loadData: ");
            readFile("data/partition_" + (device_id - 1) + "_train.txt", true);
            readFile("data/partition_" + (device_id - 1) + "_test.txt", false);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void readFile(String path, boolean isTraining) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open(path)));
        String line;
        int i = 0;
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            i++;
            String[] parts = line.trim().split(",");
            String sampleClass = parts[0].equals("1") ? "malicious" : "benign";
            float[] features = new float[TransferLearningModelWrapper.INPUT_DIM];
            for (int k = 0; k < features.length; k++) {
                features[k] = Float.parseFloat(parts[k + 1]);
            }
            try {
                this.tlModel.addSample(features, sampleClass, isTraining).get();
            } catch (ExecutionException e) {
                throw new RuntimeException("Failed to add sample to model", e.getCause());
            } catch (InterruptedException e) {
                // no-op
            }
        }
        Log.e(TAG, i + (isTraining ? " training" : " test") + " samples loaded");
        reader.close();
    }

    // function to write to a file :

    public void writeStringToFile( Context context , String fileName, String content) {
        try {
            // Get the app-specific external storage directory
            File directory = context.getExternalFilesDir(null);

            if (directory != null) {
                File file = new File(directory, fileName);

                // Check if the file exists
                boolean fileExists = file.exists();

                // Open a FileWriter in append mode
                FileWriter writer = new FileWriter(file, true);

                // If the file exists and is not empty, add a new line
                if (fileExists && file.length() > 0) {
                    writer.append("\n");
                }

                // Write the string to the file
                writer.append(content);

                // Close the FileWriter
                writer.close();
            }
        } catch (IOException e) {
            e.printStackTrace(); // Handle the exception as needed
        }
    }


}
