import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Model Once (Cache)
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_best_stack.joblib")

model = load_model()

# -------------------------
# 2. Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Titanic Survival Predictor 🚢", layout="centered", page_icon="📊")
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details and get survival prediction with confidence + visual insights.")

# -------------------------
# 3. User Inputs
# -------------------------
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 29)
sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.slider("Ticket Fare", 0.0, 520.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
title = st.selectbox("Passenger Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
deck = st.selectbox("Cabin Deck", ["A","B","C","D","E","F","G","T","Unknown"])

# -------------------------
# 4. Feature Engineering
# -------------------------
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_per_person = fare / family_size if family_size > 0 else fare

# Fixed bins (from training quantiles)
fare_bins = [0, 7.91, 14.454, 31.0, 520.0]  
fare_bin = pd.cut([fare], bins=fare_bins, labels=False, include_lowest=True)[0]

age_bin = pd.cut([age], bins=[0,12,18,35,50,80],
                 labels=["Child","Teen","Adult","Mature","Senior"]).astype(str)[0]

input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Title": title,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "Deck": deck,
    "FarePerPerson": fare_per_person,
    "AgeBin": age_bin,
    "FareBin": fare_bin
}])

st.subheader("🔍 Input Summary")
st.dataframe(input_data)

# -------------------------
# 5. Prediction + Visuals
# -------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    survival_label = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
    confidence = proba[prediction]

    st.subheader("📊 Prediction Result")
    st.markdown(f"**Prediction:** {survival_label}")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    # Probability Chart
    with st.expander("🔍 See Probability Chart"):
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(["Did Not Survive", "Survived"], proba, color=["red","green"])
        ax.set_ylabel("Probability")
        ax.set_ylim([0,1])
        for i, v in enumerate(proba):
            ax.text(i, v+0.02, f"{v:.2%}", ha="center", fontweight="bold")
        st.pyplot(fig)

    # Global Reference Chart
    with st.expander("🌍 See Global Survival Distribution"):
        global_survival = [0.62, 0.38]  # adjust if you have actual dataset ratio
        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.bar(["Did Not Survive", "Survived"], global_survival, color=["red","green"], alpha=0.6)
        ax2.set_ylabel("Proportion")
        ax2.set_ylim([0,1])
        for i, v in enumerate(global_survival):
            ax2.text(i, v+0.02, f"{v:.2%}", ha="center", fontweight="bold")
        st.pyplot(fig2)

    st.success("✅ Prediction complete! Scroll up/down for details.")
