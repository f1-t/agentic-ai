"""Layout check of the poster HTML at print size, in headless Chromium.

    python poster/check_layout.py [poster/PITWALL_poster_24x36in.html]

Reports: boxes whose content is clipped (overflow hidden and content larger than the box), text
closer than 0.5 in to the trim edge, the word count, words per printed font size in pt (SVG text
measured at its rendered size, not its nominal one), the smallest size outside the reference block
and footer, and each panel's height. Works on a temporary copy next to the input and removes it.
Exit status 1 if anything is clipped, outside the safe area, or below the size floors.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("PITWALL_poster_24x36in.html")).resolve()
JS = r"""<script>
window.addEventListener('load',()=>{
 const PX_IN=96, out=[], PT=px=>px*0.75;
 document.querySelectorAll('body *').forEach(e=>{
  const cs=getComputedStyle(e);
  if(['hidden','clip'].includes(cs.overflow)||['hidden','clip'].includes(cs.overflowY)){
   if(e.scrollHeight>e.clientHeight+1||e.scrollWidth>e.clientWidth+1)
    out.push('CLIPPED <'+e.tagName.toLowerCase()+' class="'+e.className+'"> '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,50)+
      ' | content '+e.scrollWidth+'x'+e.scrollHeight+' box '+e.clientWidth+'x'+e.clientHeight+' px');}
 });
 const hist={}; let words=0, minMain=1e9, minMainTxt='', minSmall=1e9, edge=[];
 const W=document.documentElement.scrollWidth, H=document.documentElement.scrollHeight, M=0.5*PX_IN;
 const tw=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT); let n;
 while(n=tw.nextNode()){
  const t=n.textContent.trim(); if(!t) continue; const el=n.parentElement;
  if(['SCRIPT','STYLE','TITLE'].includes(el.tagName)) continue;
  let px=parseFloat(getComputedStyle(el).fontSize);
  const svg=el.closest('svg');
  if(svg){const vb=svg.viewBox.baseVal; const r=svg.getBoundingClientRect(); if(vb&&vb.width) px*=r.width/vb.width;}
  const pt=Math.round(PT(px)*10)/10; const w=t.split(/\s+/).length; words+=w;
  const k=Math.round(pt); hist[k]=(hist[k]||0)+w;
  const small=!!el.closest('.refs, footer'), r=document.createRange(); r.selectNodeContents(n);
  for(const b of r.getClientRects()){ if(b.width<1) continue;
    if(b.left<M-0.5||b.top<M-0.5||b.right>W-M+0.5||b.bottom>H-M+0.5) edge.push(t.slice(0,40)+' @ '+[b.left,b.top,b.right,b.bottom].map(x=>(x/PX_IN).toFixed(2)).join(','));}
  if(small){ if(pt<minSmall) minSmall=pt; } else if(pt<minMain){ minMain=pt; minMainTxt=t.slice(0,40); }
 }
 out.push('PAGE '+(W/PX_IN).toFixed(2)+' x '+(H/PX_IN).toFixed(2)+' in');
 out.push('WORDS '+words);
 out.push('FONT pt : words '+JSON.stringify(hist));
 out.push('SMALLEST outside references/footer: '+minMain+' pt ('+minMainTxt+')');
 out.push('SMALLEST in references/footer: '+minSmall+' pt');
 out.push('TEXT within 0.5 in of the trim: '+(edge.length?edge.join(' | '):'none'));
 document.querySelectorAll('.panel').forEach(p=>{const r=p.getBoundingClientRect();
   out.push('PANEL '+(p.querySelector('h2')||p).textContent.trim().replace(/\s+/g,' ').slice(0,34).padEnd(34)+' '+(r.height/PX_IN).toFixed(2)+' in tall, bottom at '+(r.bottom/PX_IN).toFixed(2)+' in');});
 document.querySelectorAll('.col').forEach((c,i)=>{const last=c.lastElementChild.getBoundingClientRect(), r=c.getBoundingClientRect();
   out.push('COLUMN '+(i+1)+' content ends '+((c.scrollHeight-c.clientHeight)/PX_IN).toFixed(2)+' in past its box; box bottom '+(r.bottom/PX_IN).toFixed(2)+' in');});
 const p=document.createElement('pre'); p.id='OUT'; p.textContent=out.join('\n'); document.body.appendChild(p);
});
</script></body>"""
src = path.read_text(encoding="utf-8").replace("</body>", JS, 1)
tmp = path.with_name("_check_layout.html")
tmp.write_text(src, encoding="utf-8")
try:
    chrome = os.environ.get("CHROMIUM", "chromium")
    dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--window-size=2304,3456",
                          "--virtual-time-budget=5000", "--dump-dom", tmp.as_uri()],
                         capture_output=True, text=True, timeout=180).stdout
finally:
    tmp.unlink()
m = re.search(r'<pre id="OUT">(.*?)</pre>', dom, re.S)
report = m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&") if m else "no output"
print(report)
bad = ("CLIPPED" in report or "trim: none" not in report)
mm = re.search(r"SMALLEST outside references/footer: ([\d.]+)", report)
ms = re.search(r"SMALLEST in references/footer: ([\d.]+)", report)
if (mm and float(mm.group(1)) < 16) or (ms and float(ms.group(1)) < 12):
    bad = True
sys.exit(1 if bad else 0)
