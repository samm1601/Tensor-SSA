import time
import h5py
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import os

def tSVD(tensor, rank):
    """Randomized Tensor Singular Value Decomposition"""
    unfold_tensor = tensor.reshape(tensor.shape[0], -1)
    svd = TruncatedSVD(n_components=rank, random_state=42)
    U = svd.fit_transform(unfold_tensor)
    S = np.diag(svd.singular_values_)
    VT = svd.components_
    return U, S, VT

def tProduct(U, S, VT):
    """Tensor Product"""
    result = np.dot(U, np.dot(S, VT))
    return result

def process_chunk(img_chunk, img_gt_chunk, u, w2, L, batch_size, chunk_size, save_path, start):
    W, H, B = img_chunk.shape
    indian_pines = np.pad(img_chunk, ((u, u), (u, u), (0, 0)), mode='symmetric')
    Id = np.zeros((L, W * H), dtype=int)
    
    k = 0
    for batch_start in range(0, W, batch_size):
        batch_end = min(W, batch_start + batch_size)
        
        for chunk_start in range(batch_start, batch_end, chunk_size):
            chunk_end = min(batch_end, chunk_start + chunk_size)
            Fea_cube_chunk = np.zeros((L, (chunk_end - chunk_start) * H, B), dtype=np.float32)
            chunk_k = 0
            for i in range(chunk_start, chunk_end):
                for j in range(H):
                    i1 = i + u
                    j1 = j + u
                    k += 1
                    chunk_k += 1
                    testcube = indian_pines[i1-u:i1+u+1, j1-u:j1+u+1, :]
                    m = testcube.reshape(w2, B)

                    center = m[(w2+1)//2, :]
                    NED = np.sqrt(np.sum(((m / np.linalg.norm(m, axis=1, keepdims=True)) - (center / np.linalg.norm(center)))**2, axis=1))
                    ind = np.argsort(NED)
                    index = ind[:L]
                    Id[:, k-1] = index
                    Fea_cube_chunk[:, chunk_k-1, :] = m[index, :]
            
            batch_filename = os.path.join(save_path, f'Fea_cube_batch_{start}_{chunk_start}.npy')
            batch_filename = batch_filename.replace('\\', '/')  # Ensure consistent path separator
            try:
                np.save(batch_filename, Fea_cube_chunk)
                print(f"Saved batch: {batch_filename}")
            except OSError as e:
                print(f"Failed to save batch: {batch_filename}. Error: {e}")
            del Fea_cube_chunk  # Free memory

def process_and_classify_batches(file_path, save_path, u, w2, L, batch_size, chunk_size, class_num, scaler=None, svm_model=None, rf_model=None):
    with h5py.File(file_path, 'r') as file:
        refl_dataset = file['refl']
        gt_dataset = file['gt']
        
        # Initialize lists for training and testing data
        trainVectors, trainLabels, testVectors, testLabels = [], [], [], []
        rng = np.random.default_rng()
        Sam = 0.02
        
        for start in range(0, refl_dataset.shape[0], 50):  # Smaller batches
            end = min(start + 50, refl_dataset.shape[0])
            img_chunk = refl_dataset[start:end]
            img_gt_chunk = gt_dataset[start:end]
            
            chunk_start_time = time.time()  # Start timing the chunk processing
            process_chunk(img_chunk, img_gt_chunk, u, w2, L, batch_size, chunk_size, save_path, start)
            print(f"Chunk processing time: {time.time() - chunk_start_time:.2f} seconds")  # End timing the chunk processing
            
            # Load processed batches
            batch_files = [os.path.join(save_path, f'Fea_cube_batch_{start}_{i}.npy') for i in range(0, end-start, chunk_size)]
            
            for batch_file in batch_files:
                batch_file = batch_file.replace('\\', '/')  # Ensure consistent path separator
                if not os.path.exists(batch_file):
                    print(f"File not found: {batch_file}")
                    continue
                
                try:
                    Fea_cube_batch = np.load(batch_file, mmap_mode='r')  # Use memory mapping
                except OSError as e:
                    print(f"Failed to load batch file: {batch_file}. Error: {e}")
                    continue
                
                img_gt_batch = img_gt_chunk.reshape(-1)
                Labels = img_gt_batch
                Vectors = Fea_cube_batch.transpose(1, 0, 2).reshape(-1, L * Fea_cube_batch.shape[2])
                
                # Debugging information
                print(f"Processing batch file: {batch_file}")
                print(f"Labels shape: {Labels.shape}")
                print(f"Vectors shape: {Vectors.shape}")
                
                for k in range(1, class_num + 1):
                    index = np.where(Labels == k)[0]
                    index = index[index < Vectors.shape[0]]  # Ensure indices are within bounds
                    perclass_num = len(index)
                    
                    if perclass_num == 0:
                        continue
                    
                    Vectors_perclass = Vectors[index, :]
                    c = rng.permutation(perclass_num)
                    select_train = Vectors_perclass[c[:int(np.ceil(perclass_num * Sam))], :]
                    train_index_k = index[c[:int(np.ceil(perclass_num * Sam))]]
                    trainVectors.append(select_train)
                    trainLabels.extend([k] * int(np.ceil(perclass_num * Sam)))

                    select_test = Vectors_perclass[c[int(np.ceil(perclass_num * Sam)):], :]
                    test_index_k = index[c[int(np.ceil(perclass_num * Sam)):]]
                    testVectors.append(select_test)
                    testLabels.extend([k] * (perclass_num - int(np.ceil(perclass_num * Sam))))
        
        trainVectors = np.vstack(trainVectors)
        testVectors = np.vstack(testVectors)
        
        if scaler is None:
            scaler = StandardScaler().fit(trainVectors)
        trainVectors_scaled = scaler.transform(trainVectors)
        testVectors_scaled = scaler.transform(testVectors)
        
        model_start_time = time.time()  # Start timing the model training
        if svm_model is None:
            svm_model = SVC(kernel='linear', C=1)
        svm_model.fit(trainVectors_scaled, trainLabels)
        svm_predictions = svm_model.predict(testVectors_scaled)
        
        if rf_model is None:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(trainVectors_scaled, trainLabels)
        rf_predictions = rf_model.predict(testVectors_scaled)
        print(f"Model training and prediction time: {time.time() - model_start_time:.2f} seconds")  # End timing the model training
        
        # Metrics
        svm_accuracy = accuracy_score(testLabels, svm_predictions)
        svm_kappa = cohen_kappa_score(testLabels, svm_predictions)
        rf_accuracy = accuracy_score(testLabels, rf_predictions)
        rf_kappa = cohen_kappa_score(testLabels, rf_predictions)

        print(f"SVM Accuracy: {svm_accuracy*100:.2f}%")
        print(f"SVM Kappa: {svm_kappa*100:.2f}%")
        print(f"RF Accuracy: {rf_accuracy*100:.2f}%")
        print(f"RF Kappa: {rf_kappa*100:.2f}%")

        # Confusion Matrix
        svm_cm = confusion_matrix(testLabels, svm_predictions)
        rf_cm = confusion_matrix(testLabels, rf_predictions)

        # Plotting the confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(svm_cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0].set_title('SVM Confusion Matrix')
        axes[1].imshow(rf_cm, interpolation='nearest', cmap=plt.cm.Greens)
        axes[1].set_title('RF Confusion Matrix')
        plt.show()

        # Benchmark comparison
        benchmark_accuracy = 0.90  # 90% benchmark accuracy

        def plot_benchmark_comparison(svm_accuracy, rf_accuracy, benchmark_accuracy):
            labels = ['SVM', 'RF', 'Benchmark']
            accuracies = [svm_accuracy * 100, rf_accuracy * 100, benchmark_accuracy * 100]

            fig, ax = plt.subplots()
            ax.bar(labels, accuracies, color=['blue', 'green', 'red'])
            ax.set_ylim(0, 100)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Model Accuracy Comparison')
            for i, v in enumerate(accuracies):
                ax.text(i, v + 1, f"{v:.2f}%", ha='center')
            plt.show()

        plot_benchmark_comparison(svm_accuracy, rf_accuracy, benchmark_accuracy)
        
        return scaler, svm_model, rf_model

def main():
    # Define parameters
    u = 5
    w = 2 * u + 1
    w2 = w * w
    L = 49
    batch_size = 4  # Smaller batch size
    chunk_size = 2  # Smaller chunk size
    save_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/batches"

    # Create directory for saving batches
    os.makedirs(save_path, exist_ok=True)

    #file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/004-02reflectance.mat"
    #file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/005-01reflectance.mat"
    #file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/007-01reflectance.mat"
    #file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/008-01reflectance.mat"
    #file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/008-02reflectance.mat"
    file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/004-02reflectance.mat"
    #file_path = "C:/Users/MrLaptop/Desktop/tensor ssa/brain images/dataset/012-01reflectance.mat"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with h5py.File(file_path, 'r') as file:
        gt_dataset = file['gt']
        img_gt = gt_dataset[:]
        class_num = int(np.max(img_gt) - np.min(img_gt))

    overall_start_time = time.time()  # Start timing the overall process
    scaler, svm_model, rf_model = process_and_classify_batches(file_path, save_path, u, w2, L, batch_size, chunk_size, class_num)
    overall_elapsed_time = time.time() - overall_start_time  # End timing the overall process

    # Format overall elapsed time to minutes and seconds
    mins, secs = divmod(overall_elapsed_time, 60)
    print(f"Overall process time: {int(mins)} minutes and {secs:.2f} seconds")

if __name__ == "__main__":
    main()
