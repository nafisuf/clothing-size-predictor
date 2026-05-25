# train_model.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print(">>> Membuat data sintetis untuk training...")

# 1. Generate dataset 119.000 sampel
np.random.seed(42)
n_samples = 119000

height = np.random.uniform(140, 210, n_samples)   # cm
weight = np.random.uniform(35, 140, n_samples)    # kg
age    = np.random.randint(18, 81, n_samples)     # tahun

bmi = weight / ((height/100) ** 2)

# 2. Aturan ukuran berdasarkan BMI
def assign_size(b):
    if b < 17: return 'XXS'
    elif b < 19: return 'XS'
    elif b < 22: return 'S'
    elif b < 27: return 'M'
    elif b < 32: return 'L'
    elif b < 38: return 'XL'
    else: return 'XXL'

y = [assign_size(bmi[i]) for i in range(n_samples)]

# Tambahkan sedikit noise (5% salah label) agar tidak overfit
size_order = ['XXS','XS','S','M','L','XL','XXL']
noise = np.random.random(n_samples) < 0.05
for i in range(n_samples):
    if noise[i]:
        current = y[i]
        others = [s for s in size_order if s != current]
        y[i] = np.random.choice(others)

X = np.column_stack([height, weight, age])

# 3. Scaling dan training KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
knn.fit(X_scaled, y)

# 4. Simpan model ke folder src/models/
joblib.dump(knn, 'src/models/knn_model.pkl')
joblib.dump(scaler, 'src/models/scaler.pkl')
np.save('src/models/classes.npy', np.array(size_order))

print("✅ Model selesai dilatih dan disimpan di src/models/")
print("   - knn_model.pkl")
print("   - scaler.pkl")
print("   - classes.npy")