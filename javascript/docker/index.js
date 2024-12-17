import Docker from "dockerode";

const docker = new Docker();

// const images = await docker.listImages({
//     filters: {
//         reference: ['faasit-spilot']
//     }
// })

// const image_tags = images.flatMap(image => image.RepoTags)

// console.log(image_tags.at(-1))

// await docker.buildImage()

const image = await docker.getImage('192.168.28.220:5000/library/chammeleon-stage0:tmp')
console.log(image)