import abc

import pandas as pd
from sklearn.model_selection import train_test_split

from vkr.llm import saiga_llama3
from vkr.utils import google

TEST_SIZE = 0.2


class BaseDataset(abc.ABC):
    def __init__(self, llm_prompts: list[str] | None):
        self.dataframe = None
        self.llm_prompts = llm_prompts

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._original_init = cls.__init__
        cls.__init__ = cls.__new_init

    def __new_init(self, *args, **kwargs):
        BaseDataset.__init__(self, *args)
        self._original_init(*args, **kwargs)
        self.__post_init()

    def __post_init(self) -> None:
        assert (self.dataframe.columns == ['title', 'text', 'label']).all(), self.dataframe.columns

    def get_dataframe(self) -> pd.DataFrame:
        df = self.dataframe.copy()
        if self.llm_prompts is not None:
            inputs = []
            labels = []
            for prompt in self.llm_prompts:
                google_search = pd.Series([
                    google.SearchResult.extract_from_json(jsn).to_bert_input()
                    for jsn in google.serper_search([
                        (
                            saiga_llama3.get_query(title, text, prompt, in_db=True)
                            if prompt != 'simple'
                            else f'{title} {text}'[:300]
                        )
                        for title, text in zip(df['title'].values, df['text'].values)
                    ], None)[0]
                ])
                prompt_inputs = ('[CLS] ' + df['title'] + ' [SEP] ' + df[
                    'text'] + ' [SEP]' + google_search).values
                inputs.extend(prompt_inputs)
                labels.extend(df['label'].values)
            df = pd.DataFrame({'label': labels, 'input': inputs})
        else:
            df['input'] = '[CLS] ' + df['title'] + ' [SEP] ' + df['text'] + ' [SEP]'
        df.drop(columns=['title', 'text'], inplace=True)
        assert (df.columns == ['label', 'input']).all(), df.columns
        return df

    def get_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.get_dataframe()
        return train_test_split(df, test_size=TEST_SIZE, random_state=42)
