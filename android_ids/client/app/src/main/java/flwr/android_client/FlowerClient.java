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

    private String datasetForDevice(int deviceId) throws IOException {
        switch (deviceId) {
            case 2:
                return "mobile";
            case 3:
                return "panel";
            default:
                throw new IOException("Invalid device ID: " + deviceId);
        }
    }

    /**
     * IDS adaptation: reads assets/data/partition_<id>_train.txt and _test.txt.
     * Each line: label,f1,...,f11  (label 0 = benign, 1 = malicious; features already scaled
     * by scale_features() in dataset_load_tpot.py). Created by export_partitions.py.
     */
    public void loadData(int device_id) throws IOException {

        String dataset = datasetForDevice(device_id);

        Log.d("FLOWERCLIENT_LOAD", "Loading " + dataset + " dataset");

        readFile(
            "data/partition_" + dataset + "_train.txt",
            true
        );

        readValidationFile(
            "data/partition_" + dataset + "_val.txt"
        );

        Log.d("FLOWERCLIENT_LOAD", "Datasets loaded");

    }

    public void loadValidationData(int device_id) throws IOException {
        String dataset = datasetForDevice(device_id);

        readValidationFile(
            "data/partition_" + dataset + "_val.txt"
        );
    }

    private void readFile(String path, boolean isTraining) throws IOException {
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open(path)));
        String line;
        int i = 0;

        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            i++;
            String[] parts = line.trim().split(",");

            if (parts.length != TransferLearningModelWrapper.INPUT_DIM + 1) {
                throw new IOException(
                    "Wrong feature count in " + path +
                    ": expected " + TransferLearningModelWrapper.INPUT_DIM +
                    ", got " + (parts.length - 1)
                );
            }

            int label = Integer.parseInt(parts[0]);

            if (label != 0 && label != 1) {
                throw new IOException("Invalid label: " + label);
            }

            if (isTraining && label != 0) {
                throw new IOException(
                    "Malicious sample found in AE training data: " + path
                );
            }

            float[] features = new float[TransferLearningModelWrapper.INPUT_DIM];

            for (int k = 0; k < features.length; k++) {
                features[k] = Float.parseFloat(parts[k + 1]);
            }
            try {
                this.tlModel.addSample(features, "feature_0", isTraining).get();
            } catch (ExecutionException e) {
                throw new RuntimeException("Failed to add sample to model", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while loading " + path, e);
            }
        }
        Log.e(TAG, i + (isTraining ? " training" : " test") + " samples loaded");
        reader.close();
    }

    private void readValidationFile(String path) throws IOException {

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(path)))) {

            String line;

            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] parts = line.trim().split(",");

                if (parts.length != TransferLearningModelWrapper.INPUT_DIM + 1) {
                    throw new IOException(
                            "Wrong feature count in " + path
                    );
                }

                if (Integer.parseInt(parts[0]) != 0) {
                    throw new IOException(
                            "Non-benign validation sample in " + path
                    );
                }

                float[] features =
                        new float[TransferLearningModelWrapper.INPUT_DIM];

                for (int i = 0; i < features.length; i++) {
                    features[i] = Float.parseFloat(parts[i + 1]);
                }

                try {
                    tlModel.addValidationSample(features).get();

                } catch (ExecutionException e) {
                    throw new IOException(
                            "Failed to load validation sample", e
                    );

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException(
                            "Validation loading interrupted", e
                    );
                }
            }
        }
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
