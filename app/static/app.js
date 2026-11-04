const $ = id => document.getElementById(id);
let file = null, previewUrl = null;
async function health() {
  try { const response = await fetch('/health'); const data = await response.json();
    $('health').textContent = data.yolo_available ? 'Hybrid model ready' : 'Classical fallback ready';
  } catch { $('health').textContent = 'Service unavailable'; }
}
function selectFile(selected) {
  if (!selected) return;
  file = selected;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('preview').src = previewUrl; $('filename').textContent = file.name;
  $('drop-zone').hidden = true; $('preview-wrap').hidden = false;
  $('analyze-button').disabled = false; $('message').className = '';
  $('message').textContent = 'Image ready. Run analysis to inspect possible gaps.';
  $('output-wrap').hidden = true; $('empty-output').hidden = false;
  $('diagnostics').hidden = true; $('result-mode').textContent = 'AWAITING ANALYSIS';
  ['products','gaps','area','latency'].forEach(id => $(id).textContent = '—');
}
$('image-input').addEventListener('change', e => selectFile(e.target.files[0]));
$('change-image').addEventListener('click', () => $('image-input').click());
['dragenter','dragover'].forEach(name => $('drop-zone').addEventListener(name, e => { e.preventDefault(); $('drop-zone').classList.add('dragging'); }));
$('drop-zone').addEventListener('dragleave', () => $('drop-zone').classList.remove('dragging'));
$('drop-zone').addEventListener('drop', e => { e.preventDefault(); $('drop-zone').classList.remove('dragging'); if(e.dataTransfer.files.length) { $('image-input').files = e.dataTransfer.files; selectFile(e.dataTransfer.files[0]); } });
$('analyze-form').addEventListener('submit', async e => {
  e.preventDefault(); if(!file) return;
  $('analyze-button').disabled = true; $('image-input').disabled = true; $('change-image').disabled = true;
  $('message').className = ''; $('message').textContent = 'Analyzing texture and product occupancy…';
  try {
    const body = new FormData(); body.append('image', file);
    const response = await fetch('/api/analyze', { method:'POST', body });
    const data = await response.json(); if(!response.ok) throw new Error(data.error || 'Analysis failed');
    $('output').src = data.annotated_image_url; $('download').href = data.annotated_image_url;
    $('empty-output').hidden = true; $('output-wrap').hidden = false;
    $('result-mode').textContent = data.mode === 'hybrid' ? 'HYBRID / YOLOv8 + CV' : 'CLASSICAL FALLBACK';
    $('products').textContent = data.products_detected ?? 'N/A';
    $('product-note').textContent = data.yolo_available ? 'Localized product boxes' : 'Product model unavailable';
    $('gaps').textContent = data.gap_count; $('raw-gaps').textContent = `${data.raw_gap_count} raw texture candidates`;
    $('area').textContent = `${data.estimated_empty_area_percentage.toFixed(1)}%`;
    $('latency').textContent = `${data.processing_time_ms.toFixed(0)} ms`;
    $('message').textContent = data.yolo_available ? 'Analysis complete. Review amber regions as possible gaps.' : 'Analysis complete using texture only. Product suppression is unavailable.';
    $('json').textContent = JSON.stringify(data,null,2); $('diagnostics').hidden = false; health();
  } catch(error) { $('message').textContent = error.message; $('message').className = 'error'; }
  finally { $('analyze-button').disabled = false; $('image-input').disabled = false; $('change-image').disabled = false; }
});
health();
