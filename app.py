image_size = 128

# Label map (match your training)
label_map = {
    "fogsmog": 0, "frost": 1, "glaze": 2, "hail": 3,
    "lightning": 4, "rainbow": 5, "rainy": 6, "rime": 7, "snow": 8
}
label_map = {v: k for k, v in label_map.items()}  # Reverse it

# Genre mapping
weather_genre_map = {
    "rainy": ["jazz", "blues", "lofi"],
    "snow": ["acoustic", "classical", "ambient"],
    "rainbow": ["indie pop", "synthwave", "electronic"],
    "rime": ["chillout", "neo-classical", "dream pop"],
    "lightning": ["rock", "electro", "drum and bass"],
    "hail": ["metal", "industrial", "techno"],
    "glaze": ["trip-hop", "deep house", "minimal"],
    "frost": ["instrumental", "piano", "downtempo"],
    "fogsmog": ["ambient", "chillstep", "lofi hip hop"]
}

# Load the model (adjust path if needed)
model_path = 'weather_genre_model.keras'
model = tf.keras.models.load_model('/content/drive/MyDrive/Dataset/weather_genre_model.keras')

# =================== UI ===================
st.title("🌦️ Weather-to-Music Genre Classifier 🎧")

uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) * 2.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    predicted_weather = label_map[predicted_label]
    genres = weather_genre_map.get(predicted_weather, ["No genre found"])

    # Display result
    st.subheader(f"🌀 Detected Weather: `{predicted_weather.capitalize()}`")
    st.markdown("🎵 **Recommended Genres:**")
    for genre in genres:
        st.markdown(f"- {genre}")
