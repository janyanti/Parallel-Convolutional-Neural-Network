class sample_t {
  private:
    float *x;
    int *y;
}

enum layer_type {SIGM, SOFT, TANH, RELU, CONV};

class model {
  private:
    int num_layers;
    int *num_units; 
    layer_type *layer_types;
    // each layer has a 2d set of weights
    // float ***weights;
    float learning_rate;
    int num_epochs;

    
    int input_rows; 
    int input_cols; // should be 1 
    int output_rows; 
    int output_cols; // should be 1
    sample_t *samples;
    
  public: 
    void train(sample_t *samples, int num_samples, int input_rows, int input_cols, int output_rows, int output_cols);
    float** linearForward(float** v, float **w);
    float** linearBackward();
    float** sigmForward(float** v);
    float** sigmBackward();
    float** tanhForward(float** v);
    float** tanhBackward();
    float** reluForward(float** v);
    float** reluBackward();
    float** softForward(float** v);
    float** softBackward(float** v);
    float crossEntropyForward(float** v, float** vh);
    float crossEntropyBackward();
}     


void model::train(sample_t *samples, int num_samples, int input_rows, 
                  int input_cols, int output_rows, int output_cols) 
{
  // iterators 
  int e, s, i;
  // temp calculations 
  float ***linearComp;
  float ***activationComp;
  float ***gradLinear;
  float ***gradActivation;
  float ***gradWeights; 
  // alloc weights 
  float ***weights;
  int rows, cols;
  for (i = 0; i < num_layers; i++) {
    int rows = num_units[i];
    if (i == num_layers - 1) {
      rows = output_rows;
    } else {
      rows = num_units[i];
    }

    if (i == 0) {
      cols = input_rows;
    } else {
      cols = num_units[i-1] + 1;
    }

    weights[i] = float_alloc_2d(rows, cols);
  }

  for (e = 0; e < num_epochs; e++) {
    for (s = 0; s < num_samples; s++) {
      sample_t s = samples[s];
      // forward computation
      for (i = 0; i < num_layers; i++) {
        if (layer_types[i] != CONV)
        {
          temp[i] = linearForward(sample.x, weights[i]);
          if (layer_types[i] == SIGM)
            final[i] = sigmForward(temp[i]);
          else if (layer_types[i] == SOFT)
            final[i] = softForward(temp[i]);
        }
      }
      float J = crossEntropyForward(samples.y, final[num_layers-1]);
      float gJ = 1.;
      gradActivation[num_layers - 1] = crossEntropyBackward();
      // backward computation
      for (i = num_layers - 1; i > 1; i--) {
        if (layer_types[i] == SIGM) { 
          gradLinear[i] = sigmBackward();
          
        } else if (layer_types[i] == SOFT) {
          gradLinear[i] = softBackward();
        }
        gradWeights[i], gradActivation[i-1] = linearBackward();
      }

      gradWeights[0], gx = linearBackward();

      // update all the weights
      for (i = 0; i < num_layers; i++) {
        weights[i] -= learning_rate * gradWeights[i];
      }
    }
  }
}