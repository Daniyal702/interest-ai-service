# import numpy as np
# from sklearn.cluster import KMeans
# import joblib
# from categories import categories

# # Simulated dummy training data: 100 users, 13 category click counts (values between 0 and 10)
# np.random.seed(42)
# X_train = np.random.randint(0, 10, size=(500, len(categories)))

# # Define KMeans model (3 clusters here as example — tune as needed)
# kmeans = KMeans(n_clusters=3, random_state=42)

# # Fit model
# kmeans.fit(X_train)

# # Save model to file
# joblib.dump(kmeans, 'kmeans_model.pkl')

# print("✅ KMeans model trained and saved to model/kmeans_model.pkl")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from categories import categories

np.random.seed(42)
X_train = np.random.randint(0, 10, size=(500, len(categories)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# kmeans = KMeans(n_clusters=3, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42)

kmeans.fit(X_train_scaled)

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved.")

