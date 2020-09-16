import tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import tfnode from '@tensorflow/tfjs-node';
import fs from 'fs';

// code to import the Image
const ReadImage = (path:any) => {
  const image = fs.readFileSync(path);
  //tfnode.node.decodeImage() : Given the encoded bytes of an image, 
  //it returns a 3D or 4D tensor of the decoded image.Supports BMP, GIF, JPEG and PNG formats.
  const tfimage = tfnode.node.decodeImage(image);

  return tfimage;
  
}
//Classify image
//Image Classification function is asynchronous as it will 
//read the image,
//load the model, 
//classify it and then show the results
const ClassifyImage=async (path:any) => {
  const image = ReadImage(path);
  //MobileNets are small, low-latency, 
  //low - power models parameterized to meet the resource constraints of a variety of use cases
  const MobileNetModel = await mobilenet.load();
  //@ts-ignore
  const prediction = await MobileNetModel.classify(image);
  console.log("Classification Results:", prediction);
}
//run it as npm start <image file>
if (process.argv.length !== 3) throw new Error('Incorrect arguments: ts-node src/classify.ts <IMAGE_FILE>');

ClassifyImage(process.argv[2]);