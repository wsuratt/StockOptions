# StockOptions  
## Predicting when it is profitable to do a "straddle" on stocks in the S&P 500 index using deep learning
### Instructions for running:  
To run this project:
1. Unzip data files
2. Install required packages: pip3 install -r requirements.txt
3. Run process_data.py
4. Run model.py
5. Run market_test.py

*NOTE: A pretrained model and processed data are already included in the repo, so steps 3 and 4 are not required.*

### Processed Data:  
Symbol,ExpirationDate,CallPremium,PutPremium,Volatility,BEUp,BEDown,Days,Output
### Model Training Results:
Model Loss             |  Model Accuracy             |  Compared Accuracy
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](https://github.com/wsuratt/StockOptions/blob/main/results/model_loss.png)  |  ![alt text](https://github.com/wsuratt/StockOptions/blob/main/results/model_acc.png)  |  ![alt text](https://github.com/wsuratt/StockOptions/blob/main/results/accuracy_results.png)
### Market Testing Results:
Performance Graph            |  Final Investments
:-------------------------:|:-------------------------:
![alt text](https://github.com/wsuratt/StockOptions/blob/main/results/market_results_graph.png)  |  ![alt text](https://github.com/wsuratt/StockOptions/blob/main/results/model_results_final.png)  |  ![alt text]
