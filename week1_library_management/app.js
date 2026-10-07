// Week 1: Library Management System | Parth Denge | PRN 240705201018
const KEY = "parth_library_books";

const seed = [
  { id: 1, title: "Introduction to Algorithms", author: "T. Cormen", genre: "Technology", issued: false },
  { id: 2, title: "A Brief History of Time", author: "Stephen Hawking", genre: "Science", issued: true },
  { id: 3, title: "The Alchemist", author: "Paulo Coelho", genre: "Fiction", issued: false },
  { id: 4, title: "Python Crash Course", author: "Eric Matthes", genre: "Technology", issued: false },
  { id: 5, title: "Sapiens", author: "Yuval Noah Harari", genre: "History", issued: true },
];

let books = load();

function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || seed.slice(); }
  catch (e) { return seed.slice(); }
}
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(books)); } catch (e) { /* storage unavailable */ }
}

const $ = (id) => document.getElementById(id);

function render() {
  const q = $("search").value.trim().toLowerCase();
  const f = $("filter").value;
  const visible = books.filter((b) =>
    (b.title + " " + b.author).toLowerCase().includes(q) &&
    (f === "all" || (f === "issued") === b.issued)
  );

  $("rows").innerHTML = visible.map((b, i) => `
    <tr>
      <td>${i + 1}</td><td>${esc(b.title)}</td><td>${esc(b.author)}</td><td>${esc(b.genre)}</td>
      <td><span class="badge ${b.issued ? "issued" : "available"}">${b.issued ? "Issued" : "Available"}</span></td>
      <td>
        <button class="small" onclick="toggle(${b.id})">${b.issued ? "Return" : "Issue"}</button>
        <button class="small danger" onclick="removeBook(${b.id})">Delete</button>
      </td>
    </tr>`).join("");
  $("empty").hidden = visible.length > 0;

  const issued = books.filter((b) => b.issued).length;
  $("stats").innerHTML = `
    <div class="stat"><b>${books.length}</b>Total Books</div>
    <div class="stat"><b>${books.length - issued}</b>Available</div>
    <div class="stat"><b>${issued}</b>Issued</div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toggle(id) {
  const b = books.find((x) => x.id === id);
  if (b) { b.issued = !b.issued; save(); render(); }
}

function removeBook(id) {
  books = books.filter((x) => x.id !== id);
  save(); render();
}

$("bookForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const nextId = books.reduce((m, b) => Math.max(m, b.id), 0) + 1;
  books.push({ id: nextId, title: $("title").value.trim(), author: $("author").value.trim(), genre: $("genre").value, issued: false });
  e.target.reset();
  save(); render();
});

$("search").addEventListener("input", render);
$("filter").addEventListener("change", render);
render();
