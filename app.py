from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app,origins="*")
@app.route('/post-recommend', methods=['POST'])
def post_recommend():
    try:
        data = request.get_json()
        print("Received recommendation:", data)
        df = pd.DataFrame(data)
        
        model = joblib.load('mentor_startup_model.pkl')
        predict_df = pd.DataFrame(data) # or however you generate this

        for col in ['mentor_id', 'startup_id', 'startup_name', 'rated']:
            if col in predict_df.columns:
                predict_df = predict_df.drop(col, axis=1)

        for col in ['dwellTime', 'rating', 'averageBusiness', 'averageInnovation', 'averageMarket', 'averageTeam', 'overallAverageRating', 'totalMentorFeedback', 'totalViews']:
            if col in predict_df.columns:
                predict_df[col] = predict_df[col].fillna(0)
        for col in ['bookmarked', 'domain_match']:
            if col in predict_df.columns:
                predict_df[col] = predict_df[col].astype(int)
        probs = model.predict_proba(predict_df)[:, 1]  # Probability for class 1 (recommended)
        result_df = pd.DataFrame(data)
        result_df['recommendation_score'] = probs
        result_df_sorted = result_df.sort_values('recommendation_score', ascending=False)
        print(result_df_sorted[['startup_name', 'startup_id', 'recommendation_score']].head(10))
        result_df_sorted = result_df_sorted[['startup_name', 'startup_id', 'recommendation_score']]

        result_df_sorted.to_json('result_prediction.json', orient='records', lines=False)

        return jsonify({
            "message": "Recommendation received successfully",
            "data": result_df_sorted.to_dict(orient='records')
        }), 200

    except Exception as e:
        print("Error:", str(e))  # 👈 Add this
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=6000)
