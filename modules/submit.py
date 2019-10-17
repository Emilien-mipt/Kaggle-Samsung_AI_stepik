import pandas as pd

def make_submit(test_img, test_pred):
    submission_df = pd.DataFrame.from_dict({'id': test_img, 'label': test_pred})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace('test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)
    submission_df.head(n=6)
    submission_df.to_csv('./submission.csv')
