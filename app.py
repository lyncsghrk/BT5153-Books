import logging
from flask import Flask, render_template, request
from model import Model
import plotly.graph_objects as go

model = Model()
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
PRED_CACHE = dict()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test_users')
def test_users():
    model.run_predictions_on_full_test()
    model.prepare_user_details()

    # Options for the dropdown menu
    user_details = model.user_details['top_books'].to_dict()
    return render_template('test_users.html', user_details=user_details)


@app.route('/test_users/<chosen_user>')
def process(chosen_user):
    # Get book recommendations
    if chosen_user in PRED_CACHE:
        preds_df = PRED_CACHE[chosen_user]
    else:
        preds_df = model.get_user_predictions(chosen_user)
        PRED_CACHE[chosen_user] = preds_df

    if preds_df is None:
        return "No predictions hit!"

    # Get Pandas series of recommended books
    recommended_books = preds_df.set_index('book_id')[['title_without_series', 'target', 'final_score']]
    recommended_books['is_recommended'] = recommended_books['final_score'] >= 0.45

    # Use Bootstrap's List to make a list of recommended books and a button for each book, routing to '/explain/book_id'
    # Render the page with recommended books
    return render_template(
        'recommended_books.html',
        chosen_user=chosen_user,
        recommended_books=recommended_books
    )


@app.route('/test_users/<chosen_user>/<int:chosen_book>')
def explain(chosen_user, chosen_book):
    # Get book recommendations
    # This should be a cache hit since we're coming from `process`, but we include the else path just in case
    if chosen_user in PRED_CACHE:
        preds_df = PRED_CACHE[chosen_user]
    else:
        preds_df = model.get_user_predictions(chosen_user)
        PRED_CACHE[chosen_user] = preds_df

    # Get Pandas series of recommended books
    recommended_books = preds_df.set_index('book_id')[['title_without_series', 'target', 'final_score']]
    recommended_books['is_recommended'] = recommended_books['final_score'] >= 0.45

    # book_details = model.all_books[model.all_books['book_id'] == book_id]
    logging.info(f"Generating explanation for user:{chosen_user}, book:{chosen_book}")

    book_df = preds_df.set_index('book_id').loc[chosen_book]
    waterfall_cols = [
        'intercept',
        'clus_score',
        'gen_score',
        'desc_score',
        'rev_score',
        'user_score',
        'tit_score',
        'final_score'
    ]
    waterfall_display_cols = [
        'Intercept',
        'Book Clustering Similarity',
        'Genre Similarity',
        'Description Topic Similarity',
        'Review Vector Similarity',
        'User Clustering Similarity',
        'Title Vector Similarity',
        'Sum of Sub-Model Scores'
    ]
    waterfall_data = book_df[waterfall_cols].tolist()
    fig = go.Figure(
        go.Waterfall(
            name='Recommendation explanation',
            orientation='h',
            measure=['relative', 'relative', 'relative', 'relative', 'relative', 'relative', 'relative', 'total'],
            y=waterfall_display_cols,
            x=waterfall_data
        )
    )
    fig_html = fig.to_html(full_html=False)

    top_model_idx = waterfall_cols.index(book_df[waterfall_cols[:-1]].astype(float).idxmax())
    top_model = waterfall_display_cols[top_model_idx]
    explanation_str = f"The highest contributing model was {top_model}. "
    if book_df['final_score'] >= 0.45:
        reasons = [
            '-', # intercept
            'it is similar to books you enjoyed in terms of book statistics like popularity and page count.',
            'it is similar to books you enjoyed in terms of overlapping genres.',
            'it is similar to books you enjoyed in terms of description similarity.',
            'it is similar to books you enjoyed in terms of review similarity.',
            'other users similar to you in taste enjoyed this book.',
            'it is similar to books you enjoyed in terms of title similarity.',
        ]
        explanation_str += "This means that this book was recommended since "
        explanation_str += reasons[top_model_idx]
    else:
        explanation_str += "However, the confidence score is below the threshold of 0.45, so it is not recommended."

    score_sum = f"{sum(waterfall_data[:-1]):.5f}"
    final_score = f"{book_df['final_score']:.5f}"
    return render_template(
        'recommended_books.html',
        chosen_user=chosen_user,
        recommended_books=recommended_books,
        render_explanation='true',
        fig=fig_html,
        score_sum=score_sum,
        final_score=final_score,
        explanation_str=explanation_str
    )


if __name__ == '__main__':
    app.run(debug=True)
