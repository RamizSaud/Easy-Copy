const express = require("express");
const multer = require("multer");
const path = require("path");
const { exec } = require("child_process");
const cors = require("cors");
const app = express();
const port = 5000;

app.use(cors());

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "model/images/");
  },
  filename: (req, file, cb) => {
    cb(
      null,
      file.fieldname + "-" + Date.now() + ".png" //path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

app.post("/upload", upload.single("image"), (req, res) => {
  //   console.log("File received:", req.file);
  res.status(200).send("Image uploaded successfully!");
});

app.get("/", (req, res) => {
  const run = "python ./model/model.py";
  exec(run, (error, stdout, stderr) => {
    if (error) {
      console.error(`exec error: ${error}`);
      return;
    }
    // console.log(stdout);
    res.send({ text: stdout.trim() });
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
