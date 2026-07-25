/* ═══════════════ NeuroScan AI — App JS ═══════════════ */
(function(){
'use strict';

// ── DOM refs ──
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const previewFilename = document.getElementById('preview-filename');
const previewSize = document.getElementById('preview-size');
const previewRemove = document.getElementById('preview-remove');
const analyzeBtn = document.getElementById('analyze-btn');
const analyzeText = document.querySelector('.analyze-text');
const analyzeLoader = document.getElementById('analyze-loader');
const resultsSection = document.getElementById('results-section');
const resultsContent = document.getElementById('results-content');
const errorCard = document.getElementById('error-card');
const errorMessage = document.getElementById('error-message');
const retryBtn = document.getElementById('retry-btn');
const newAnalysisBtn = document.getElementById('new-analysis-btn');
const heatmapBtn = document.getElementById('heatmap-btn');
const heatmapBtnText = document.querySelector('.heatmap-btn-text');
const heatmapBtnLoader = document.getElementById('heatmap-btn-loader');
const heatmapResult = document.getElementById('heatmap-result');

let selectedFile = null;
let currentImageFilename = ''; // server-side filename for heatmap

// ── Particles ──
(function(){
  const bg = document.getElementById('particles-bg');
  for(let i=0;i<30;i++){
    const p = document.createElement('div');
    p.className = 'particle';
    const s = Math.random()*4+2;
    p.style.cssText = `width:${s}px;height:${s}px;left:${Math.random()*100}%;animation-duration:${Math.random()*10+8}s;animation-delay:${Math.random()*10}s;`;
    bg.appendChild(p);
  }
})();

// ── Upload Zone ──
uploadZone.addEventListener('click', ()=> fileInput.click());
uploadZone.addEventListener('dragover', e=>{e.preventDefault();uploadZone.classList.add('dragover')});
uploadZone.addEventListener('dragleave', ()=>uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e=>{
  e.preventDefault(); uploadZone.classList.remove('dragover');
  if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', ()=>{if(fileInput.files.length) handleFile(fileInput.files[0])});

function handleFile(file){
  if(!file.type.startsWith('image/')){showToast('Please select an image file','error');return}
  if(file.size>32*1024*1024){showToast('File too large (max 32MB)','error');return}
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e=>{previewImage.src=e.target.result};
  reader.readAsDataURL(file);
  previewFilename.textContent = file.name;
  previewSize.textContent = (file.size/1024).toFixed(1)+' KB';
  uploadZone.style.display='none';
  previewContainer.style.display='block';
  resultsSection.style.display='none';
}

previewRemove.addEventListener('click', resetUpload);
function resetUpload(){
  selectedFile=null; currentImageFilename='';
  fileInput.value='';
  previewContainer.style.display='none';
  uploadZone.style.display='block';
  resultsSection.style.display='none';
}



// ═══════════════════════════════════════════
// STEP 1: INSTANT CLASSIFICATION
// ═══════════════════════════════════════════
analyzeBtn.addEventListener('click', classifyImage);

async function classifyImage(){
  if(!selectedFile) return;
  analyzeBtn.disabled=true;
  analyzeText.style.display='none';
  analyzeLoader.style.display='flex';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try{
    const resp = await fetch('/predict',{method:'POST',body:formData});
    const data = await resp.json();

    analyzeBtn.disabled=false;
    analyzeText.style.display='inline';
    analyzeLoader.style.display='none';

    if(data.success){
      currentImageFilename = data.image_filename;
      showResults(data);
    } else {
      showError(data.error||'Unknown error');
    }
  }catch(err){
    analyzeBtn.disabled=false;
    analyzeText.style.display='inline';
    analyzeLoader.style.display='none';
    showError('Network error: '+err.message);
  }
}

// ═══════════════════════════════════════════
// STEP 2: OPTIONAL HEATMAP GENERATION
// ═══════════════════════════════════════════
heatmapBtn.addEventListener('click', generateHeatmap);

async function generateHeatmap(){
  if(!currentImageFilename) return;
  heatmapBtn.disabled=true;
  heatmapBtnText.style.display='none';
  heatmapBtnLoader.style.display='flex';

  try{
    const resp = await fetch('/heatmap',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image_filename: currentImageFilename})
    });
    const data = await resp.json();

    heatmapBtn.disabled=false;
    heatmapBtnText.style.display='inline';
    heatmapBtnLoader.style.display='none';

    if(data.success){
      showHeatmapResult(data);
    } else {
      showToast('Heatmap failed: '+data.error, 'error');
    }
  }catch(err){
    heatmapBtn.disabled=false;
    heatmapBtnText.style.display='inline';
    heatmapBtnLoader.style.display='none';
    showToast('Network error: '+err.message, 'error');
  }
}

function showHeatmapResult(data){
  heatmapResult.style.display='block';
  heatmapBtn.style.display='none';
  const introEl = document.querySelector('.heatmap-intro');
  if(introEl) introEl.style.display='none';

  const origImg = document.getElementById('hm-img-original');
  const heatImg = document.getElementById('hm-img-heatmap');
  origImg.src = '/uploads/'+currentImageFilename;
  heatImg.src = data.heatmap_image;

  document.getElementById('heatmap-time').textContent = 'ScoreCAM heatmap generated in '+data.processing_time+'s';

  setupImageTabs();
}

function setupImageTabs(){
  const tabs=document.querySelectorAll('.img-tab');
  const pair=document.getElementById('image-pair');
  const origBox=document.getElementById('hm-orig-box');
  const heatBox=document.getElementById('hm-heat-box');
  tabs.forEach(tab=>{
    tab.addEventListener('click',()=>{
      tabs.forEach(t=>t.classList.remove('active'));
      tab.classList.add('active');
      const mode=tab.dataset.tab;
      if(mode==='side-by-side'){
        pair.classList.remove('single');
        origBox.style.display='block'; heatBox.style.display='block';
      }else if(mode==='original'){
        pair.classList.add('single');
        origBox.style.display='block'; heatBox.style.display='none';
      }else{
        pair.classList.add('single');
        origBox.style.display='none'; heatBox.style.display='block';
      }
    });
  });
}

// ═══════════════════════════════════════════
// SHOW RESULTS (instant — no heatmap)
// ═══════════════════════════════════════════
function showResults(data){
  resultsSection.style.display='block';
  errorCard.style.display='none';
  resultsContent.style.display='block';

  // Reset heatmap section for new analysis
  heatmapResult.style.display='none';
  heatmapBtn.style.display='block';
  const introEl = document.querySelector('.heatmap-intro');
  if(introEl) introEl.style.display='block';

  // Severity
  const sevMap={high:{color:'#ef4444',bg:'rgba(239,68,68,.12)',text:'High Severity'},
                moderate:{color:'#f59e0b',bg:'rgba(245,158,11,.12)',text:'Moderate Severity'},
                low:{color:'#3b82f6',bg:'rgba(59,130,246,.12)',text:'Low Severity'},
                none:{color:'#4caf50',bg:'rgba(76,175,80,.12)',text:'Healthy Result'}};
  const sev = sevMap[data.tumor_info.severity]||sevMap.none;

  // Header
  document.getElementById('result-class').textContent = data.tumor_info.name||data.prediction;
  document.getElementById('result-fullname').textContent = data.tumor_info.full_name||'';
  const viewLabel = data.detected_view ? data.detected_view.charAt(0).toUpperCase()+data.detected_view.slice(1) : 'Unknown';
  document.getElementById('result-view').textContent = 'Detected View: '+viewLabel;
  const sevEl = document.getElementById('result-severity');
  sevEl.style.background = sev.bg; sevEl.style.color = sev.color;
  sevEl.style.border = '1px solid '+sev.color+'33';
  sevEl.querySelector('.severity-dot').style.background = sev.color;
  sevEl.querySelector('.severity-text').textContent = sev.text;
  document.getElementById('result-time').textContent = 'Classified in '+data.processing_time+'s';
  document.getElementById('result-time').style.color = 'var(--text-muted)';
  document.getElementById('result-time').style.fontSize = '.8rem';
  document.getElementById('result-time').style.marginTop = '8px';

  // Confidence ring
  const conf = data.confidence;
  const ringFill = document.getElementById('ring-fill');
  const circumference = 2*Math.PI*52;
  ringFill.style.stroke = sev.color;
  ringFill.style.strokeDashoffset = circumference; // reset
  setTimeout(()=>{ringFill.style.strokeDashoffset = circumference-(conf/100)*circumference},100);
  animateNumber('ring-value',0,Math.round(conf),1200);

  // Probability bars
  const probBars = document.getElementById('prob-bars');
  probBars.innerHTML='';
  const colors = {glioma:'linear-gradient(90deg,#ef4444,#f97316)',meningioma:'linear-gradient(90deg,#f59e0b,#eab308)',
                  notumor:'linear-gradient(90deg,#22c55e,#10b981)',pituitary:'linear-gradient(90deg,#3b82f6,#6366f1)'};
  const sorted = Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]);
  sorted.forEach(([cls,pct],i)=>{
    const item = document.createElement('div'); item.className='prob-item';
    const label = cls==='notumor'?'No Tumor':cls.charAt(0).toUpperCase()+cls.slice(1);
    item.innerHTML=`<span class="prob-label">${label}</span>
      <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%;background:${colors[cls]||'var(--gradient)'}"></div></div>
      <span class="prob-value">${pct.toFixed(1)}%</span>`;
    probBars.appendChild(item);
    setTimeout(()=>{item.querySelector('.prob-bar-fill').style.width=Math.max(pct,2)+'%'},150+i*120);
  });

  // Original image
  document.getElementById('img-original').src = data.original_image;

  // Tumor info grid
  const grid = document.getElementById('tumor-grid'); grid.innerHTML='';
  const info = data.tumor_info;
  [{label:'Description',value:info.description,full:true},{label:'WHO Grade',value:info.grade},
   {label:'Location',value:info.location},{label:'Prevalence',value:info.prevalence},
   {label:'Age Group',value:info.age_group},{label:'Survival Rate',value:info.survival_rate,full:true},
   {label:'Risk Factors',value:info.risk_factors,full:true}
  ].forEach(f=>{
    if(!f.value||f.value==='N/A')return;
    const d=document.createElement('div');d.className='tumor-field'+(f.full?' full':'');
    d.innerHTML=`<div class="tumor-field-label">${f.label}</div><div class="tumor-field-value">${f.value}</div>`;
    grid.appendChild(d);
  });

  // Symptoms
  const symList=document.getElementById('symptoms-list'); symList.innerHTML='';
  if(info.symptoms&&info.symptoms.length){
    document.getElementById('symptoms-card').style.display='block';
    info.symptoms.forEach(s=>{const li=document.createElement('li');li.textContent=s;symList.appendChild(li)});
  } else { document.getElementById('symptoms-card').style.display='none'; }

  // Treatment
  const treatList=document.getElementById('treatment-list'); treatList.innerHTML='';
  if(info.treatment&&info.treatment.length){
    document.getElementById('treatment-card').style.display='block';
    info.treatment.forEach(t=>{const li=document.createElement('li');li.textContent=t;treatList.appendChild(li)});
  } else { document.getElementById('treatment-card').style.display='none'; }

  // Tech details
  const techGrid=document.getElementById('tech-grid'); techGrid.innerHTML='';
  const mi=data.model_info, im=data.image_metadata;
  [{l:'Architecture',v:mi.architecture},{l:'Feature Dim',v:mi.feature_dim+'D'},
   {l:'Classifier',v:mi.classifier},{l:'Device',v:mi.device.toUpperCase()},
   {l:'Detected View',v:viewLabel},{l:'Classification Time',v:data.processing_time+'s'},
   {l:'Image Size',v:im.width+'x'+im.height+'px'},{l:'File Size',v:im.file_size_kb+' KB'}
  ].forEach(t=>{
    const d=document.createElement('div');d.className='tech-item';
    d.innerHTML=`<div class="tech-label">${t.l}</div><div class="tech-value">${t.v}</div>`;
    techGrid.appendChild(d);
  });

  setTimeout(()=>resultsSection.scrollIntoView({behavior:'smooth',block:'start'}),200);
}

function animateNumber(id,start,end,duration){
  const el=document.getElementById(id);
  const range=end-start; const startTime=performance.now();
  function step(ts){
    const p=Math.min((ts-startTime)/duration,1);
    el.textContent=Math.round(start+range*(1-Math.pow(1-p,3)));
    if(p<1)requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function showError(msg){
  resultsSection.style.display='block';
  errorCard.style.display='block';
  resultsContent.style.display='none';
  errorMessage.textContent=msg;
}

retryBtn.addEventListener('click',()=>{resultsSection.style.display='none';classifyImage()});
newAnalysisBtn.addEventListener('click',()=>{resetUpload();window.scrollTo({top:document.getElementById('upload-section').offsetTop-80,behavior:'smooth'})});

function showToast(msg,type){
  const t=document.createElement('div');
  t.style.cssText=`position:fixed;top:80px;right:20px;z-index:300;padding:14px 24px;border-radius:10px;font-size:.9rem;font-weight:500;
    background:${type==='error'?'rgba(239,68,68,.9)':'rgba(0,212,255,.9)'};color:#fff;backdrop-filter:blur(10px);
    animation:fadeIn .3s;font-family:'Inter',sans-serif;max-width:350px;`;
  t.textContent=msg;
  document.body.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transition='opacity .3s';setTimeout(()=>t.remove(),300)},3000);
}

})();
