import express from "express";
import {getLecture} from '../controllers/getLecture.js'
const router=express.Router();

router.get("/:lecctureId",getLecture)

export default router; 