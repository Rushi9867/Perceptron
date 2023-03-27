from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron
import pandas as pd
import logging
import os

gate = "XOR gate"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),
                    level=logging.INFO,
                    format='[%(asctime)s:%(levelname)s:%(module)s]:%(message)s',
                    filemode='a')

def main(data,modelName,plotName,eta,epochs):
    df_XOR = pd.DataFrame(XOR)

    X,y = prepare_data(df_XOR)

    model_xor = Perceptron(eta=eta,epochs=epochs)
    model_xor.fit(X,y)
    _ = model_xor.total_loss()

    model_xor.save(filename=modelName,model_dir="new_models")
    save_plot(df_XOR,model_xor,filename=plotName)

if __name__ == '__main__':
    XOR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,0]
    }
    ETA = 0.1
    EPOCHS =10
        
    try:
        logging.info(f">>>>> starting training for {gate} >>>>>")
        main(data=XOR,modelName="xor.model",plotName="xor.png",eta=ETA,epochs=EPOCHS)
        logging.info(f"<<<<< done training for {gate} <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e
