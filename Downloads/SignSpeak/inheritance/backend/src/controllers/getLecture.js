import Lecture from "../models/Lecture.js";
import cloudinary from "../config/cloudinary.js";

export async function getLecture(req, res) {
  const { lectureId } = req.params;

  const lecture = await Lecture.findById(lectureId);

  // TODO: check auth & enrollment here

  const videoUrl = cloudinary.url(lecture.videoPublicId, {
    resource_type: "video",
    secure: true
  });

  res.json({
    title: lecture.title,
    duration: lecture.duration,
    videoUrl
  });
};
