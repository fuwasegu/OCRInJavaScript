import { MnistData } from 'https://storage.googleapis.com/tfjs-tutorials/mnist_data.js';
import { showExamples, getModel, train } from './lib.js';

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;

async function run() {
    let signaturePad = initSignaturePad();

    const data = new MnistData();
    await data.load();
    await showExamples(data);

    let model = getModel();

    tfvis.show.modelSummary({ name: 'Model Architecture' }, model);

    await train(model, data);

    document.getElementById("loading-message").style.display = "none";
    document.getElementById("main").style.display = "block";

    document.getElementById("predict-button").onclick = function () {
        predict(model);
    };

    document.getElementById("reset-button").onclick = function(){
        reset(signaturePad);
    };
}

document.addEventListener('DOMContentLoaded', run);

function initSignaturePad() {
    let pad = document.getElementById('pad');
    return new SignaturePad(pad, {
        minWidth: 5,
        maxWidth: 5,
        penColor: 'white',
        backgroundColor: 'black',
    });
}

function predict(model) {
    let ctx = document.createElement('canvas').getContext('2d');
    let pad = document.getElementById('pad');
    ctx.drawImage(pad, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
    let imageData = ctx.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

    let score = tf.tidy(() => {
        let input = tf.browser.fromPixels(imageData, 1).reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]).div(tf.scalar(255));
        return model.predict(input);
    });

    let maxConfidence = score.dataSync().indexOf(Math.max.apply(null, score.dataSync()));

    let elements = document.querySelectorAll(".confidence");

    elements.forEach((el, i) => {
        el.parentNode.classList.remove('is-selected');
        if (i === maxConfidence) {
            el.parentNode.classList.add('is-selected');
        }
        el.innerText = score.dataSync()[i];
    });
}

function reset(signaturePad) {
    signaturePad.clear();
    let elements = document.querySelectorAll(".confidence");
    elements.forEach(el => {
        el.parentNode.classList.remove('is-selected');
        el.innerText = '-';
    });
}