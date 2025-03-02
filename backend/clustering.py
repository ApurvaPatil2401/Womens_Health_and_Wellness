import numpy as np

def health_recommendation(cluster):
    recommendations = {
        0: (
            "🔴 High PCOS Risk:\n"
            "- Consult a gynecologist for medical evaluation.\n"
            "- Follow a low-carb, high-fiber diet (whole grains, legumes, greens).\n"
            "- Engage in regular exercise (30 min daily - cardio & strength training).\n"
            "- Consider supplements (Vitamin D, Myo-Inositol, Omega-3s).\n"
            "- Monitor menstrual cycles & hormonal changes closely."
        ),
        1: (
            "🟠 Moderate PCOS Symptoms:\n"
            "- Improve diet: Reduce sugar & processed foods.\n"
            "- Exercise: Yoga, pilates, or brisk walking (30 min daily).\n"
            "- Manage stress with meditation, deep breathing, or therapy.\n"
            "- Track ovulation & menstrual cycles.\n"
            "- Consider natural supplements (Spearmint tea for hormone balance)."
        ),
        2: (
            "🟢 Low PCOS Risk:\n"
            "- Maintain a balanced diet with whole foods, lean protein, and healthy fats.\n"
            "- Continue regular physical activity (at least 3 times per week).\n"
            "- Keep hydrated and get 7-9 hours of sleep daily.\n"
            "- Monitor menstrual health & visit a doctor annually for checkups."
        )
    }
    
    return recommendations.get(cluster, "General lifestyle modifications recommended.")

def predict_cluster(input_data, model):
    cluster = model.predict(input_data)[0]
    recommendation = health_recommendation(cluster)
    return cluster, recommendation
