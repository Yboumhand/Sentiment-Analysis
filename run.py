# ============================================================================
# run.py - SCRIPT DE LANCEMENT DE L'APPLICATION
# ============================================================================

import os
from app import app
from config import config

# Charger la configuration appropriée
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 LANCEMENT DE L'APPLICATION FLASK")
    print("=" * 80)
    print(f"\n📊 Environnement : {env}")
    print(f"🐛 Debug mode    : {app.config['DEBUG']}")
    print(f"🔐 Secret key    : {'✅ Configurée' if app.config['SECRET_KEY'] else '❌ Non configurée'}")
    print("\n🌐 Application disponible sur :")
    print("   • http://127.0.0.1:5000")
    print("   • http://localhost:5000")
    print("\n💡 Appuyez sur Ctrl+C pour arrêter le serveur")
    print("=" * 80 + "\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )