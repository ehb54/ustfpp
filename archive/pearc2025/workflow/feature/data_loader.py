import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict

class DataLoader:
    def __init__(self, data_file: str):
        self.data = pd.read_csv(data_file, sep=',', skipinitialspace=True)
        self.features = None
        self.target = None
        self.encoded_features = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._prepare_data()

    def _prepare_data(self):
        # Split features and target
        self.features = self.data.drop(['CPUTime', 'source_file', 'directory'], axis=1, errors='ignore')
        self.target = self.data['CPUTime']
        self.encoded_features = self._encode_features()

    def _encode_features(self):
        encoded = self.features.copy()

        # Convert all string columns to categorical
        categorical_columns = encoded.select_dtypes(include=['object']).columns
        for feature in categorical_columns:
            le = LabelEncoder()
            encoded[feature] = le.fit_transform(self.features[feature].astype(str))
            self.label_encoders[feature] = le

        return encoded

    def get_qualitative_features(self) -> List[str]:
        return self.features.select_dtypes(include=['object']).columns.tolist()

    def get_quantitative_features(self) -> List[str]:
        return self.features.select_dtypes(include=['int64', 'float64']).columns.tolist()
