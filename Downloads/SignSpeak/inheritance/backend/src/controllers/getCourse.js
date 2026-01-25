import Course from "../models/Course.js";
import Section from "../models/Section.js";

export async function getCourse (req, res) {
  try{
  const { courseId } = req.params;

  const course = await Course.findById(courseId);

  const sections = await Section.find({ courseId })
    .populate("lectures", "title duration");

  res.status(200).res.json({
    _id: course._id,
    title: course.title,
    description: course.description,
    sections
  });
}
catch(error){
  console.log("Error in getCourse controller ",error);
  res.status(500).json({message:"Internal server error"});
}
};
