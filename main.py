import tensorflow as tf
from trainer import Trainer
from tester import LimeTester
def testLimeExplanations():
    limeTester=LimeTester()
    limeTester.runLimeExplanations()

def trainDataset():
    hgt=Trainer()
    hgt.train_model()

def main():
    print("Running Main Method")
    trainDataset()
    #testLimeExplanations()
   
if __name__ == "__main__":
    main()

