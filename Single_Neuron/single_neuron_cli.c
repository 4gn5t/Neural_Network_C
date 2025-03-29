#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "main.h"

SingleNeuronClassifier* create_neuron(int num_features, ActivationFunction activation) {
    SingleNeuronClassifier* neuron = (SingleNeuronClassifier*)malloc(sizeof(SingleNeuronClassifier));
    
    neuron->num_features = num_features;
    neuron->weights = (double*)calloc(num_features, sizeof(double));
    neuron->delta_weights_prev = (double*)calloc(num_features, sizeof(double));
    neuron->bias = 0.0;
    neuron->delta_bias_prev = 0.0;
    neuron->activation = activation;
    
    neuron->log_capacity = 1000;
    neuron->log_size = 0;
    neuron->epochs = (int*)malloc(neuron->log_capacity * sizeof(int));
    neuron->train_errors = (double*)malloc(neuron->log_capacity * sizeof(double));
    neuron->val_errors = (double*)malloc(neuron->log_capacity * sizeof(double));
    
    initialize_weights(neuron);
    
    return neuron;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

double relu_activation(double x) {
    return fmax(0.0, x);
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double softmax_actication(double x) {
    return exp(x) / (1.0 + exp(x));
}

double softmax_derivative(double x) {
    double s = softmax_actication(x);
    return s * (1.0 - s);
}

void initialize_weights(SingleNeuronClassifier* neuron) {
    for (int i = 0; i < neuron->num_features; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    }
    neuron->bias = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
}

double forward(SingleNeuronClassifier* neuron, double* x) {
    double z = neuron->bias;
    for (int i = 0; i < neuron->num_features; i++) {
        z += neuron->weights[i] * x[i];
    }
    
    switch (neuron->activation) {
        case SIGMOID:
            return sigmoid(z);
        case TANH:
            return tanh_activation(z);
        case RELU:
            return relu_activation(z);
        case SOFTMAX:
            return softmax_actication(z);
        default:
            return sigmoid(z);
    }
}

double compute_error(int* y_true, double* y_pred, int n) {
    double sum_error = 0.0;
    for (int i = 0; i < n; i++) {
        double mse = y_true[i] - y_pred[i];
        sum_error += mse * mse;
    }
    return sum_error / n;
}

void add_to_log(SingleNeuronClassifier* neuron, int epoch, double train_error, double val_error) {
    if (neuron->log_size >= neuron->log_capacity) {
        neuron->log_capacity *= 2;
        neuron->epochs = (int*)realloc(neuron->epochs, neuron->log_capacity * sizeof(int));
        neuron->train_errors = (double*)realloc(neuron->train_errors, neuron->log_capacity * sizeof(double));
        neuron->val_errors = (double*)realloc(neuron->val_errors, neuron->log_capacity * sizeof(double));
    }
    
    neuron->epochs[neuron->log_size] = epoch;
    neuron->train_errors[neuron->log_size] = train_error;
    neuron->val_errors[neuron->log_size] = val_error;
    neuron->log_size++;
}

int gradient_descent(SingleNeuronClassifier* neuron, Dataset* data, double alpha, double nu, double E0, int max_epochs) {
    double* y_pred_train = (double*)malloc(data->num_train * sizeof(double));
    double* y_pred_val = (double*)malloc(data->num_val * sizeof(double));
    
    double train_error = 1.0;
    double val_error = 1.0;
    int epoch = 0;
    
    while (epoch < max_epochs && train_error > E0) {
        // forward pass
        for (int i = 0; i < data->num_train; i++) {
            y_pred_train[i] = forward(neuron, data->X_train[i]);
        }
        
        // error
        train_error = compute_error(data->y_train, y_pred_train, data->num_train);
        
        // backward pass
        for (int i = 0; i < neuron->num_features; i++) {
            double gradient = 0.0;
            
            for (int j = 0; j < data->num_train; j++) {
                double x_ji = data->X_train[j][i];
                double error = data->y_train[j] - y_pred_train[j];
                
                double z = neuron->bias;
                for (int k = 0; k < neuron->num_features; k++) {
                    z += neuron->weights[k] * data->X_train[j][k];
                }
                
                double derivative;
                switch (neuron->activation) {
                    case SIGMOID: derivative = sigmoid_derivative(z); break;
                    case TANH: derivative = tanh_derivative(z); break;
                    case RELU: derivative = relu_derivative(z); break;
                    case SOFTMAX: derivative = softmax_derivative(z); break;
                    default: derivative = sigmoid_derivative(z);
                }
                
                gradient += error * derivative * x_ji;
            }
            
            gradient /= data->num_train;
            
            double delta_w = alpha * gradient + nu * neuron->delta_weights_prev[i];
            neuron->weights[i] += delta_w;
            neuron->delta_weights_prev[i] = delta_w;
        }
        
        double bias_gradient = 0.0;
        for (int j = 0; j < data->num_train; j++) {
            double error = data->y_train[j] - y_pred_train[j];
            double z = neuron->bias;
            for (int k = 0; k < neuron->num_features; k++) {
                z += neuron->weights[k] * data->X_train[j][k];
            }
            
            double derivative;
            switch (neuron->activation) {
                case SIGMOID: derivative = sigmoid_derivative(z); break;
                case TANH: derivative = tanh_derivative(z); break;
                case RELU: derivative = relu_derivative(z); break;
                case SOFTMAX: derivative = softmax_derivative(z); break;
                default: derivative = sigmoid_derivative(z);
            }
            
            bias_gradient += error * derivative;
        }
        
        bias_gradient /= data->num_train;
        double delta_b = alpha * bias_gradient + nu * neuron->delta_bias_prev;
        neuron->bias += delta_b;
        neuron->delta_bias_prev = delta_b;
        
        for (int i = 0; i < data->num_val; i++) {
            y_pred_val[i] = forward(neuron, data->X_val[i]);
        }
        val_error = compute_error(data->y_val, y_pred_val, data->num_val);
        
        add_to_log(neuron, epoch, train_error, val_error);
        
        if (epoch % 10 == 0) {
            printf("Epoch %d: train_error=%.6f, val_error=%.6f\r", epoch, train_error, val_error);
            fflush(stdout);
        }
        
        epoch++;
    }
    
    printf("\n");  
    
    free(y_pred_train);
    free(y_pred_val);
    
    return epoch;
}

void predict(SingleNeuronClassifier* neuron, double** X, int n, int* predictions, double threshold) {
    for (int i = 0; i < n; i++) {
        double output = forward(neuron, X[i]);
        
        if (neuron->activation == TANH) {
            predictions[i] = output >= 0 ? 1 : 0; 
        } else if (neuron->activation == SOFTMAX) {
            predictions[i] = output >= threshold ? 1 : 0; 
        } else if (neuron->activation == RELU) {
            predictions[i] = output >= threshold ? 1 : 0; 
        } else if (neuron->activation == SIGMOID) {
            predictions[i] = output >= threshold ? 1 : 0; 
        } else {
            predictions[i] = output >= threshold ? 1 : 0; 
        }
    }
}

double compute_accuracy(int* y_true, int* y_pred, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return (double)correct / n;
}

void normalize_data(Dataset* data) {
    data->mean = (double*)calloc(data->num_features, sizeof(double));
    data->std_dev = (double*)calloc(data->num_features, sizeof(double));
    
    for (int j = 0; j < data->num_features; j++) {
        for (int i = 0; i < data->num_samples; i++) {
            data->mean[j] += data->X[i][j];
        }
        data->mean[j] /= data->num_samples;
    }
    
    for (int j = 0; j < data->num_features; j++) {
        for (int i = 0; i < data->num_samples; i++) {
            double diff = data->X[i][j] - data->mean[j];
            data->std_dev[j] += diff * diff;
        }
        data->std_dev[j] = sqrt(data->std_dev[j] / data->num_samples);
        
        if (data->std_dev[j] < 1e-8) {
            data->std_dev[j] = 1.0;
        }
    }
    
    for (int i = 0; i < data->num_samples; i++) {
        for (int j = 0; j < data->num_features; j++) {
            data->X[i][j] = (data->X[i][j] - data->mean[j]) / data->std_dev[j];
        }
    }
}

void split_data(Dataset* data, double train_ratio, double val_ratio, double test_ratio) {
    train_ratio = 0.7;  //  70%
    val_ratio = 0.2;    //  20%
    test_ratio = 0.1;  // 10%
    
    data->num_train = (int)(data->num_samples * train_ratio);
    data->num_val = (int)(data->num_samples * val_ratio);
    data->num_test = data->num_samples - data->num_train - data->num_val;
    
    int total_assigned = data->num_train + data->num_val + data->num_test;
    if (total_assigned < data->num_samples) {
        data->num_train += (data->num_samples - total_assigned);
    }
    
    printf("Training set: %d samples (%.1f%%)\n", data->num_train, (double)data->num_train / data->num_samples * 100);
    printf("Validation set: %d samples (%.1f%%)\n", data->num_val,(double)data->num_val / data->num_samples * 100);
    printf("Testing set: %d samples (%.1f%%)\n", data->num_test, (double)data->num_test / data->num_samples * 100);
    printf("\n");
    
    data->X_train = (double**)malloc(data->num_train * sizeof(double*));
    data->y_train = (int*)malloc(data->num_train * sizeof(int));
    data->X_val = (double**)malloc(data->num_val * sizeof(double*));
    data->y_val = (int*)malloc(data->num_val * sizeof(int));
    data->X_test = (double**)malloc(data->num_test * sizeof(double*));
    data->y_test = (int*)malloc(data->num_test * sizeof(int));
    
    int* indices = (int*)malloc(data->num_samples * sizeof(int));
    for (int i = 0; i < data->num_samples; i++) {
        indices[i] = i;
    }
    
    for (int i = 0; i < data->num_samples; i++) {
        int j = rand() % data->num_samples;
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    int idx = 0;
    
    for (int i = 0; i < data->num_train; i++) {
        int sample_idx = indices[idx++];
        data->X_train[i] = data->X[sample_idx];
        data->y_train[i] = data->y[sample_idx];
    }
    
    for (int i = 0; i < data->num_val; i++) {
        int sample_idx = indices[idx++];
        data->X_val[i] = data->X[sample_idx];
        data->y_val[i] = data->y[sample_idx];
    }
    
    for (int i = 0; i < data->num_test; i++) {
        int sample_idx = indices[idx++];
        data->X_test[i] = data->X[sample_idx];
        data->y_test[i] = data->y[sample_idx];
    }
    
    free(indices);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    char* command = argv[1];
    
    if (strcmp(command, "all") == 0) {
        if (argc < 4) {
            printf("Usage: %s all <data_file> <activation_function_to_save>\n", argv[0]);
            return 1;
        }        
        
        const char* data_file = argv[2];
        char* active_function_save = argv[3];

        Dataset* data = load_data(data_file, 1, 2);
        
        print_data_info(data);
        normalize_data(data);
        
        char* activation_names[4] = {"SIGMOID", "TANH", "RELU", "SOFTMAX"};
        ActivationFunction activations[4] = {SIGMOID, TANH, RELU, SOFTMAX};
        
        clock_t funct_activations_start = clock();
        for (int i = 0; i < 4; i++) {
            split_data(data, 0.7, 0.2, 0.1);
            printf("\n\n==== Training with %s activation function ====\n", activation_names[i]);
            
            SingleNeuronClassifier* neuron = create_neuron(data->num_features, activations[i]);
            
            clock_t start = clock();
            printf("Epochs: %d\n", 2000);
            printf("Alpha: %f\n", 0.01);
            printf("Nu: %f\n", 0.0);
            printf("E0: %f\n", 0.01);
            printf("Max epochs: %d\n", 2000);
            int epochs = gradient_descent(neuron, data, 0.01, 0.0, 0.01, 2000);
            clock_t end = clock();
            double training_time = (double)(end - start) / CLOCKS_PER_SEC;
            
            int* train_preds = (int*)malloc(data->num_train * sizeof(int));
            predict(neuron, data->X_train, data->num_train, train_preds, 0.5);
            double train_accuracy = compute_accuracy(data->y_train, train_preds, data->num_train);
            
            int* val_preds = (int*)malloc(data->num_val * sizeof(int));
            predict(neuron, data->X_val, data->num_val, val_preds, 0.5);
            double val_accuracy = compute_accuracy(data->y_val, val_preds, data->num_val);
            
            int* test_preds = (int*)malloc(data->num_test * sizeof(int));
            predict(neuron, data->X_test, data->num_test, test_preds, 0.5);
            double test_accuracy = compute_accuracy(data->y_test, test_preds, data->num_test);
            
            printf("\n=== TRAINING RESULTS ===\n");
            printf("Training completed in %d epochs\n", epochs);
            printf("Final train error: %f\n", neuron->train_errors[neuron->log_size-1]);
            printf("Final validation error: %f\n", neuron->val_errors[neuron->log_size-1]);
            printf("Training time: %.2f seconds\n", training_time);

            printf("\n=== ACCURACY EVALUATION ===\n");
            printf("Train accuracy: %.2f%%\n", train_accuracy * 100);
            printf("Validation accuracy: %.2f%%\n", val_accuracy * 100);
            printf("Test accuracy: %.2f%%\n", test_accuracy * 100);
            
            printf("\n=== MODEL PARAMETERS ===\n");
            print_neuron_info(neuron);

            if (strcmp(activation_names[i], active_function_save) == 0) {
                save_training_history(neuron, "training_history.csv");
                save_predictions_to_csv(data->y_train, train_preds, data->X_train, data->num_train, data->num_features, "train_predictions.csv");
                save_predictions_to_csv(data->y_val, val_preds, data->X_val, data->num_val, data->num_features, "val_predictions.csv");
                save_predictions_to_csv(data->y_test, test_preds, data->X_test, data->num_test, data->num_features, "test_predictions.csv");    
            }
            free(train_preds);
            free(val_preds);
            free(test_preds);
            free_neuron(neuron);

        }
        

        clock_t funct_activations_end = clock();
        double funct_activations_time = (double)(funct_activations_end - funct_activations_start) / CLOCKS_PER_SEC;
        printf("Total time for all activations: %.2f seconds\n", funct_activations_time);
        
        free_dataset(data);

    }
    else {
        print_usage();
    }
    
    return 0;
}

