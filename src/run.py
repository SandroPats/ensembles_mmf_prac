import shutil
import os
from ml_server import app

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'static/tmp/')
    if os.path.exists(path):
        shutil.rmtree(path)
    app.run(host='0.0.0.0', port=5000, debug=True)
