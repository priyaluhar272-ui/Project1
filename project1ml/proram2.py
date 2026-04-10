import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("music_dataset_1200.csv")

X = df[['tempo','energy','loudness','danceability','valence']]
y = df['mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print("\nEnter Song Details 👇")
tempo = float(input("Tempo (50-150): "))
energy = float(input("Energy (0-1): "))
loudness = float(input("Loudness (-20 to -3): "))
dance = float(input("Danceability (0-1): "))
valence = float(input("Valence (0-1): "))

new_data = pd.DataFrame([[tempo, energy, loudness, dance, valence]],
columns=['tempo','energy','loudness','danceability','valence'])
result = model.predict(new_data)[0]

print("\n🎧 Music Mood Detection Result 🎧")
print("----------------------------------")
print(f"Prediction: {result}")

if result == 1:
    print("Mood: Happy 😊")
else:
    print("Mood: Sad 😢")

print(f"Accuracy: {accuracy:.2f}")
print("----------------------------------")
print("Model Status: Working Successfully ✅")