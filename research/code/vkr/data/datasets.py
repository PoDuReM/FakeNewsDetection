import pathlib

import pandas as pd

from vkr.data import preprocessing
from vkr.data.base_dataset import BaseDataset
from vkr.utils import translate


class Welfake(BaseDataset):
    def __init__(self, csv_path: pathlib.Path | str, add_google: bool = False) -> None:
        super(Welfake, self).__init__(None if not add_google else ['simple'])
        csv_path = pathlib.Path(csv_path)
        df = pd.read_csv(csv_path)[['title', 'text', 'label']]
        self.dataframe = preprocessing.welfake(df)


class FakeNewsPredictions(BaseDataset):
    def __init__(self, csv_path: pathlib.Path | str, add_google: bool = False) -> None:
        super(FakeNewsPredictions, self).__init__(None if not add_google else ['simple'])
        csv_path = pathlib.Path(csv_path)
        df = pd.read_csv(csv_path)[['title', 'text', 'label']]
        self.dataframe = preprocessing.fake_news_prediction(df)


class RussianWelfake(BaseDataset):
    def __init__(self, csv_path: pathlib.Path | str, llm_prompts: list[str] | None = None) -> None:
        super(RussianWelfake, self).__init__(llm_prompts)
        csv_path = pathlib.Path(csv_path)
        df = pd.read_csv(csv_path)[['russian_title', 'russian_text', 'label']].rename(
            columns={'russian_title': 'title', 'russian_text': 'text'},
        )
        self.dataframe = preprocessing.welfake(df)

    @staticmethod
    def translate_and_save(
            original_welfake_path: pathlib.Path | str,
            path_to_save_russian: pathlib.Path | str,
            max_workers: int = 12,
    ) -> bool:
        ok = True
        dts = Welfake(original_welfake_path)
        dts.dataframe = dts.dataframe
        russian_titles = translate.apply_translation_multithreaded(
            dts.dataframe,
            'title',
            max_workers=max_workers,
        )
        dts.dataframe['russian_title'] = russian_titles
        ids = [translate.FAIL_STRING not in title for title in russian_titles]
        if not all(ids):
            ok = False
        dts.dataframe = dts.dataframe.iloc[ids]
        russian_texts = translate.apply_translation_multithreaded(
            dts.dataframe,
            'text',
            max_workers=max_workers,
        )
        dts.dataframe['russian_text'] = russian_texts
        ids = [translate.FAIL_STRING not in text for text in russian_texts]
        if not all(ids):
            ok = False
        dts.dataframe = dts.dataframe.iloc[ids]
        dts.dataframe.to_csv(path_to_save_russian, index=False)
        return ok


class RussianKaggle(BaseDataset):
    REAL_CSV_NAMES = ['dw', 'insider', 'meduza', 'novaja', 'zona']
    FAKE_CSV_NAMES = ['panorama']

    def __init__(self, folder_path: pathlib.Path | str, llm_prompts: list[str] | None = None) -> None:
        super(RussianKaggle, self).__init__(llm_prompts)
        folder_path = pathlib.Path(folder_path)
        dfs = []
        for csv_name in RussianKaggle.REAL_CSV_NAMES + RussianKaggle.FAKE_CSV_NAMES:
            label = int(csv_name in RussianKaggle.REAL_CSV_NAMES)
            df = pd.read_csv(folder_path / f'{csv_name}.csv', header=None).dropna()
            if len(df.columns) == 2:
                tts = df[0].apply(self.__extract_title_text)
                dfs.append(pd.DataFrame({
                    'title': [x[0] for x in tts],
                    'text': [x[1] for x in tts],
                    'label': [label] * len(df),
                }))
            else:
                assert len(df.columns) == 3
                df.rename(columns={0: 'title', 1: 'text'}, inplace=True)
                df.drop(columns=[2], inplace=True)
                df['label'] = [label] * len(df)
                dfs.append(df)
        df = pd.concat(dfs)
        self.dataframe = preprocessing.russian_kaggle(df)

    @staticmethod
    def __extract_title_text(text: str) -> tuple[str, str]:
        spl = text.split('.')
        return spl[0], '.'.join(spl[1:])
