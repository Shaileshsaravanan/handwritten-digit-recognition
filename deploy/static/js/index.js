var canvas = document.getElementById('canvas');
canvas.width = 280;
canvas.height = 280;

var context = canvas.getContext('2d');
context.fillStyle = 'black';
context.fillRect(0, 0, canvas.width, canvas.height);

var range = document.getElementById('customRange1');
range.oninput = function () {
    this.title = 'lineWidth: ' + this.value;
};

var Mouse = { x: 0, y: 0 };
var lastMouse = { x: 0, y: 0 };
var painting = false;

canvas.onmousedown = function () {
    painting = !painting;
};

canvas.onmousemove = function (e) {
    lastMouse.x = Mouse.x;
    lastMouse.y = Mouse.y;
    Mouse.x = e.pageX - this.offsetLeft;
    Mouse.y = e.pageY - this.offsetTop;
    if (painting) {
        context.lineWidth = range.value;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.strokeStyle = 'white';
        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.closePath();
        context.stroke();
    }
};

canvas.onmouseup = function () {
    painting = !painting;
};

var predict = document.getElementById('predict');
predict.onclick = function () {
    var canvas = document.getElementById('canvas');
    imgUrl = canvas.toDataURL('image/jpeg');
    $.ajax({
        type: 'POST',
        url: '/predict/',
        data: imgUrl,
        success: function (r) {
            $('#result').text('Prediction: ' + r.prediction);
            $('#result').attr('data-original-title', 'Confidence: ' + r.confidence);
        },
        error: function (e) {
            console.log(e);
        }
    });
};

var clear = document.getElementById('clear');
clear.onclick = function () {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
    $('#result').html('Prediction:&nbsp;&nbsp;&nbsp;');
    $('#result').attr('data-original-title', '');
};

var start_x, start_y, move_x, move_y, end_x, end_y;

canvas.ontouchstart = function (e) {
    start_x = e.touches[0].pageX - this.offsetLeft;
    start_y = e.touches[0].pageY - this.offsetTop;
    context.lineWidth = range.value;
    context.lineJoin = 'round';
    context.lineCap = 'round';
    context.strokeStyle = 'white';
    context.beginPath();
    context.moveTo(start_x, start_y);
};

canvas.ontouchmove = function (e) {
    move_x = e.touches[0].pageX - this.offsetLeft;
    move_y = e.touches[0].pageY - this.offsetTop;
    context.lineTo(move_x, move_y);
    context.stroke();
};

canvas.ontouchend = function (e) {
    end_x = e.changedTouches[0].pageX - this.offsetLeft;
    end_y = e.changedTouches[0].pageY - this.offsetTop;
    context.closePath();
};

$(function () {
    $('[data-toggle="tooltip"]').tooltip();
});