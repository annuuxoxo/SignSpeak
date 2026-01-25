import mongoose from "mongoose";

const lectureSchema = new mongoose.Schema({
  courseId: { type: mongoose.Schema.Types.ObjectId, ref: "Course" },
  sectionId: { type: mongoose.Schema.Types.ObjectId, ref: "Section" },
  title: {
    type:String,
    required:true,
  },
  duration: String,
  videoPublicId: String,   // Cloudinary ID
});
const Lecture=mongoose.model("Lecture", lectureSchema);
export default Lecture;
