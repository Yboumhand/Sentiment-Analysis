# ============================================================================
# app.py - APPLICATION FLASK PRINCIPALE
# ============================================================================

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from datetime import datetime
import json
import os

# Initialisation de l'application
app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'

# Charger le modèle et les objets nécessaires
print("📦 Chargement du modèle...")

try:
    # Charger le modèle optimisé
    with open('models/final_optimized_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
        model = model_package['model']
        model_name = model_package['model_name']
        test_metrics = model_package['test_metrics']

    # Charger le vectorizer et le label encoder
    with open('data/raw/processed/vectorization_objects.pkl', 'rb') as f:
        vect_objects = pickle.load(f)
        vectorizer = vect_objects['tfidf_vectorizer']
        label_encoder = vect_objects['label_encoder']
        scaler = vect_objects['scaler']
        additional_feature_cols = vect_objects['additional_feature_cols']

    print(f"✅ Modèle chargé : {model_name}")
    print(f"✅ Accuracy : {test_metrics['accuracy']:.2%}")

except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    model = None

# Stockage de l'historique (en production, utiliser une base de données)
predictions_history = []


# Fonction pour extraire les features additionnelles
def extract_features(text):
    """Extrait les features additionnelles d'un texte"""
    from textblob import TextBlob

    if not text:
        return np.zeros(len(additional_feature_cols))

    words = text.split()

    # Features basiques
    char_count = len(text)
    word_count = len(words)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    exclamation_count = text.count('!')
    question_count = text.count('?')

    # Sentiment avec TextBlob
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except:
        polarity = 0
        subjectivity = 0

    features = np.array([
        char_count,
        word_count,
        avg_word_length,
        exclamation_count,
        question_count,
        polarity,
        subjectivity
    ])

    # Normaliser
    features_scaled = scaler.transform(features.reshape(1, -1))

    return features_scaled.flatten()


def predict_sentiment(text):
    """Prédit le sentiment d'un texte"""
    if not model or not text.strip():
        return None

    # Vectoriser le texte
    text_vectorized = vectorizer.transform([text])

    # Extraire features additionnelles
    additional_features = extract_features(text)

    # Combiner les features
    from scipy.sparse import hstack, csr_matrix
    features_combined = hstack([
        text_vectorized,
        csr_matrix(additional_features.reshape(1, -1))
    ])

    # Prédiction
    prediction = model.predict(features_combined)[0]

    # Probabilités (si disponible)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_combined)[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = None

    # Décoder le label
    sentiment = label_encoder.inverse_transform([prediction])[0]

    return {
        'sentiment': sentiment,
        'prediction_code': int(prediction),
        'confidence': confidence,
        'text': text,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html',
                           model_name=model_name,
                           accuracy=f"{test_metrics['accuracy']:.2%}",
                           f1_score=f"{test_metrics['f1_score']:.2%}")


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour prédire le sentiment"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400

        # Faire la prédiction
        result = predict_sentiment(text)

        if result:
            # Ajouter à l'historique
            predictions_history.append(result)

            # Limiter l'historique à 100 entrées
            if len(predictions_history) > 100:
                predictions_history.pop(0)

            return jsonify(result)
        else:
            return jsonify({'error': 'Erreur lors de la prédiction'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Endpoint pour prédire plusieurs textes"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'Aucun texte fourni'}), 400

        results = []
        for text in texts:
            if text.strip():
                result = predict_sentiment(text)
                if result:
                    results.append(result)

        return jsonify({'predictions': results, 'count': len(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Page d'historique des prédictions"""
    return render_template('history.html',
                           predictions=predictions_history[::-1],
                           model_name=model_name,
                           accuracy=f"{test_metrics['accuracy']:.2%}")  # Plus récent en premier


@app.route('/api/history')
def api_history():
    """API pour obtenir l'historique"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        'predictions': predictions_history[-limit:][::-1],
        'total': len(predictions_history)
    })


@app.route('/stats')
def stats():
    """Page de statistiques"""
    if not predictions_history:
        stats_data = {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'avg_confidence': 0
        }
    else:
        positive_count = sum(1 for p in predictions_history if p['sentiment'] == 'Positif')
        negative_count = sum(1 for p in predictions_history if p['sentiment'] == 'Négatif')

        confidences = [p['confidence'] for p in predictions_history if p['confidence'] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0

        stats_data = {
            'total': len(predictions_history),
            'positive': positive_count,
            'negative': negative_count,
            'positive_pct': (positive_count / len(predictions_history) * 100) if predictions_history else 0,
            'negative_pct': (negative_count / len(predictions_history) * 100) if predictions_history else 0,
            'avg_confidence': avg_confidence
        }

    return render_template('stats.html', stats=stats_data, model_name=model_name,
                           accuracy=f"{test_metrics['accuracy']:.2%}")


@app.route('/api/stats')
def api_stats():
    """API pour obtenir les statistiques"""
    if not predictions_history:
        return jsonify({
            'total': 0,
            'positive': 0,
            'negative': 0,
            'avg_confidence': 0
        })

    positive_count = sum(1 for p in predictions_history if p['sentiment'] == 'Positif')
    negative_count = sum(1 for p in predictions_history if p['sentiment'] == 'Négatif')

    confidences = [p['confidence'] for p in predictions_history if p['confidence'] is not None]
    avg_confidence = float(np.mean(confidences)) if confidences else 0

    return jsonify({
        'total': len(predictions_history),
        'positive': positive_count,
        'negative': negative_count,
        'positive_pct': (positive_count / len(predictions_history) * 100),
        'negative_pct': (negative_count / len(predictions_history) * 100),
        'avg_confidence': avg_confidence
    })


@app.route('/about')
def about():
    """Page À propos"""
    return render_template('about.html',
                           model_name=model_name,
                           metrics=test_metrics, 
                           accuracy=f"{test_metrics['accuracy']:.2%}")


# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 LANCEMENT DE L'APPLICATION")
    print("=" * 80)
    print(f"\n📊 Modèle : {model_name}")
    print(f"🎯 Accuracy : {test_metrics['accuracy']:.2%}")
    print(f"🎯 F1-Score : {test_metrics['f1_score']:.2%}")
    print("\n🌐 Application disponible sur : http://127.0.0.1:5000")
    print("\n" + "=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)