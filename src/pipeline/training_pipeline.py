from src.components.data_ingestion import Ingestion
from src.components.data_preprocessing import Preprocessing
from src.components.model_trainer import Model_Training

class Train_Pipeline():
  def start_training_pipeline(self):
    reply = input('Do your wish to Perform Hyperparamter Tunning Y/n')
    if reply == 'n' or reply=='N' :
      # Step 1: Ingestion
      ingestion_process = Ingestion(file = 'data\dataset_full.csv')
      dir = ingestion_process.start_ingestion()

      # Step 2: Preprocessing
      preprocess_process = Preprocessing(cleaned_data=dir['cleaned_data'],train_data=dir['train_data'], test_data=dir['test_data']) 
      data = preprocess_process.preprocessing()

      # Step 3: Model Training
      model_training_process = Model_Training(data['X'], data['y'],dir['test_data'],dir['cleaned_data'])
      model_dir = model_training_process.start_training(model = None)
      return model_dir

    else:
      # Step 1: Ingestion
      ingestion_process = Ingestion(file = 'data\dataset_full.csv')
      dir = ingestion_process.start_ingestion()

      # Step 2: Preprocessing
      preprocess_process = Preprocessing(cleaned_data=dir['cleaned_data'],train_data=dir['train_data'], test_data=dir['test_data']) 
      data = preprocess_process.preprocessing()

      # Step 3: Model Training
      model_training_process = Model_Training(data['X'], data['y'],dir['test_data'],dir['cleaned_data'])
      model = model_training_process.start_hyperparameter_tunning()
      model_dir = model_training_process.start_training(model=model)
      return model_dir

       

if __name__ == "__main__":
    # Create an instance of Train_Pipeline
    train_pipeline_instance = Train_Pipeline()

    # Start the training pipeline
    pipeline_dir = train_pipeline_instance.start_training_pipeline()
    print(f"Pipeline trained and saved to: {pipeline_dir}")