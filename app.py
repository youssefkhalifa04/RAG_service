from flask import Flask, jsonify
from flask_cors import CORS
from model.Qwen3 import Qwen3
from services.knowledge_generator import KnowledgeGenerator
from services.report_generator import ReportGenerator
from storage.SupabaseStorage import SupabaseStorage
from model.gemini import GeminiModel
import os
app = Flask(__name__)
CORS(app)

model = GeminiModel(os.getenv("GOOGLE_API_KEY") , model_id="gemini-3-flash" , embed_model_id="models/text-embedding-004")
#model = Qwen3()
supabase = SupabaseStorage()
report_generator = ReportGenerator(supabase, model)
knowledge_generator = KnowledgeGenerator(supabase, model) 

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        knowledge_generator.generate_knowledge()
        return jsonify({"message": "Summarization process completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/report", methods=["POST"])
def report():
    try:
        print("Generating report...") 
        
        text = report_generator.generate_summary(type="employee", factory_id="97e90fd2-469a-471b-a824-1e6ac0d5ec93" , query="who is the best performer", top_k=5 )
        print("Report generated:", text)
        return jsonify({"message": "Report generated successfully.", "report": text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
