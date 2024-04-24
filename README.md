## Stock Price Prediction with LSTM Model

This code implements a Long Short-Term Memory (LSTM) model to predict the closing price of Berkshire Hathaway stock (BRK-B) using historical data.

**Features:**

* Uses historical closing and opening prices to calculate the daily price change.
* You can add more features (replace 'Change' with your desired features)

**Process:**

1. **Data Download:** Downloads BRK-B stock data from Yahoo Finance for the period 2000-01-01 to 2024-12-31 using `yfinance`.
2. **Feature Engineering:** Calculates the daily price change (`Change`) as the difference between closing and opening prices.
3. **Data Preprocessing:**
    * Scales the data between 0 and 1 using `MinMaxScaler` from scikit-learn.
    * Defines a sequence length (number of past days to consider for prediction). Experiment with different sequence lengths.
    * Creates sequences of data for training and testing using the `create_sequences` function.
4. **Model Building:**
    * Builds an LSTM model with multiple LSTM layers, Dropout layers, and a final Dense layer for prediction.
    * Compiles the model with the Adam optimizer and mean squared error (MSE) loss function.
5. **Training:** Trains the model on the training data for a specified number of epochs and batch size.
6. **Evaluation:**
    * Evaluates the model's performance on the test data using MSE.
    * Prints the Mean Squared Error.
7. **Saving:** Saves the trained model as `Stock_Prediction_Model.h5`.

**How to Use:**

1. Download and install the required libraries: `numpy`, `pandas`, `yfinance`, `tensorflow`, `scikit-learn`.
2. Run the script.
3. The trained model will be saved as `Stock_Prediction_Model.h5`.  You can use this model for further predictions on new data (similar to the code for making predictions on new data is commented out).

**Notes:**

* This is a basic example. Stock price prediction is a complex task, and there's no guaranteed way to achieve perfect accuracy.
* Experiment with different hyperparameters (number of units, dropout rates, etc.) and features to improve the model's performance.

I hope this README file is helpful!
