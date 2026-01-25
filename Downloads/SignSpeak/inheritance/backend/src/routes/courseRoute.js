import express from "express";
import {getCourse} from '../controllers/getCourse.js'
const router=express.Router();

router.get("/:courseId",getCourse)

export default router;