// Week 2: Canvas Whiteboard | Parth Denge | PRN 240705201018
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const colorEl = document.getElementById("color");
const sizeEl = document.getElementById("size");

let tool = "pen";
let drawing = false;
let start = null;
let currentStroke = null;
const strokes = [];   // committed shapes
const redoStack = [];

function pos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (canvas.width / r.width), y: (e.clientY - r.top) * (canvas.height / r.height) };
}

function drawShape(s) {
  ctx.lineWidth = s.size;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = s.tool === "eraser" ? "#ffffff" : s.color;
  ctx.beginPath();
  if (s.tool === "pen" || s.tool === "eraser") {
    s.points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  } else if (s.tool === "line") {
    ctx.moveTo(s.a.x, s.a.y); ctx.lineTo(s.b.x, s.b.y);
  } else if (s.tool === "rect") {
    ctx.rect(s.a.x, s.a.y, s.b.x - s.a.x, s.b.y - s.a.y);
  } else if (s.tool === "circle") {
    ctx.arc(s.a.x, s.a.y, Math.hypot(s.b.x - s.a.x, s.b.y - s.a.y), 0, Math.PI * 2);
  }
  ctx.stroke();
}

function redraw() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  strokes.forEach(drawShape);
  if (currentStroke) drawShape(currentStroke);
}

canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  start = pos(e);
  currentStroke = { tool, color: colorEl.value, size: +sizeEl.value, points: [start], a: start, b: start };
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = pos(e);
  currentStroke.points.push(p);
  currentStroke.b = p;
  redraw();
});

function finish() {
  if (!drawing) return;
  drawing = false;
  strokes.push(currentStroke);
  currentStroke = null;
  redoStack.length = 0;
  redraw();
}
canvas.addEventListener("pointerup", finish);
canvas.addEventListener("pointerleave", finish);

document.querySelectorAll("[data-tool]").forEach((btn) =>
  btn.addEventListener("click", () => {
    tool = btn.dataset.tool;
    document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("active", b === btn));
  })
);

document.getElementById("undo").onclick = () => { if (strokes.length) { redoStack.push(strokes.pop()); redraw(); } };
document.getElementById("redo").onclick = () => { if (redoStack.length) { strokes.push(redoStack.pop()); redraw(); } };
document.getElementById("clear").onclick = () => { strokes.length = 0; redoStack.length = 0; redraw(); };
document.getElementById("save").onclick = () => {
  const a = document.createElement("a");
  a.download = "parth_denge_whiteboard.png";
  a.href = canvas.toDataURL("image/png");
  a.click();
};

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "z") document.getElementById("undo").click();
  if (e.ctrlKey && e.key === "y") document.getElementById("redo").click();
});

// Starter sketch so the board is not empty on first load.
strokes.push(
  { tool: "rect", color: "#2f6fed", size: 5, points: [], a: { x: 120, y: 110 }, b: { x: 420, y: 300 } },
  { tool: "circle", color: "#dc2626", size: 5, points: [], a: { x: 640, y: 210 }, b: { x: 740, y: 210 } },
  { tool: "line", color: "#16a34a", size: 6, points: [], a: { x: 120, y: 420 }, b: { x: 860, y: 420 } }
);
redraw();
