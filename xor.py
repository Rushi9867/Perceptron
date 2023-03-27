from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron
import pandas as pd

def main(data,modelName,plotName,eta,epochs):
    df_XOR = pd.DataFrame(XOR)

    X,y = prepare_data(df_XOR)

    model_xor = Perceptron(eta=eta,epochs=epochs)
    model_xor.fit(X,y)
    _ = model_xor.total_loss()

    model_xor.save(filename=modelName,model_dir="new_model_xor")
    save_plot(df_XOR,model_xor,filename=plotName)

if __name__ == '__main__':
    XOR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,0]
    }
    ETA = 0.1
    EPOCHS =10
    main(data=XOR,modelName="xor.model",plotName="new_xor.png",eta=ETA,epochs=EPOCHS)
