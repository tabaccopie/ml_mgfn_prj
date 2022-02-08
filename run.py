import pickle
import pandas as pd


class predict_t():
    """Predict target"""

    def __init__(self, path_pipeline: str, path_result: str) -> None:
       
        self.path_pipeline = path_pipeline
        self.file_out_result = path_result

        # Load pipeline
        with open(self.path_pipeline, 'rb') as file:
            self.model = pickle.load(file)

    def run(self, path_data: str) -> None:
       
        df_init = pd.read_csv(path_data)

        df = pd.DataFrame()
        df[['id', 'vas_id', 'buy_time']] = df_init[['id', 'vas_id', 'buy_time']]
        df['target'] = self.model.predict_proba(df_init)[:,1]
        df.to_csv(self.file_out_result)


if __name__ == '__main__':

    newmodel = predict_t(
        path_pipeline='final_model.pkl',
        path_result='answers_test.csv'
    )
    newmodel.run(path_data='data_test.csv')