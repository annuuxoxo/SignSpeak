import express from "express";
import cors from "cors";
import courseRoute from "./routes/courseRoute.js";
import lectureRoute from "./routes/lectureRoute.js";
import authRoute from "./routes/authRoute.js";
import { connectDB } from "./config/db.js";
import dotenv from "dotenv";

dotenv.config();

const app=express();
const PORT=process.env.PORT || 5001;

connectDB();

app.use(cors());
app.use(express.json());
app.use("/api/auth", authRoute);
app.use("/api/courses",courseRoute);
app.use("/api/lectures",lectureRoute);

app.listen(5001,()=>{
    console.log("Server started on Port", PORT);
});