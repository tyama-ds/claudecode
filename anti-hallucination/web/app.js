'use strict';
const $ = id => document.getElementById(id);
let token = '', samples = null, jobId = null, currentResult = null, currentRunId = null;
let running = false, pollFailures = 0;
const labels = {correct:'正答',incorrect:'誤答',abstain:'保留',error:'失敗',supported:'支持',refuted:'誤り',insufficient:'保留',completed:'完了',cancelled:'停止',failed:'失敗',running:'実行中'};
const pct = v => v === null || v === undefined ? '—' : `${(100 * v).toFixed(1)}%`;
const num = v => v === null || v === undefined ? '—' : Number(v).toLocaleString('ja-JP',{maximumFractionDigits:2});
const money = v => v === null || v === undefined ? (currentResult?.run_mode==='recorded_agent_pilot'?'未計測':'未設定') : `$${Number(v).toFixed(4)}`;
const displayDate = value => {const date=new Date(value);return value&&!Number.isNaN(date.getTime())?date.toLocaleString('ja-JP',{timeZone:'Asia/Tokyo'})+' JST':value||'';};
const runMode = r => r.run_mode === 'recorded_agent_pilot' ? 'LLM回答取込・同一モデル' : r.demo ? 'デモ' : '実API';
const el = (tag,text,cls) => {const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
function showError(message){$('error').textContent=message;$('error').hidden=false;}
function clearError(){$('error').hidden=true;}
async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Pilot-Token':token},body:JSON.stringify(body)};
  const res=await fetch(path,options);const data=await res.json();
  if(!res.ok)throw new Error(data.error||`HTTP ${res.status}`);return data;
}
function inputs(){
  const values={};for(const id of ['reports','gold','config']){try{values[id]=JSON.parse($(id).value);}catch(err){throw new Error(`${{reports:'レポート',gold:'正解データ',config:'モデル設定'}[id]}のJSONを確認してください: ${err.message}`);}}
  return values;
}
function setInput(id,value,name){$(id).value=JSON.stringify(value,null,2);$('name-'+id).textContent=name;}
function updateMode(){
  try{const cfg=JSON.parse($('config').value);const demo=cfg.models?.every(m=>m.provider==='demo');$('mode-note').className='notice'+(demo?'':' live');$('mode-note').textContent=demo?'デモモード：架空の資料と模擬モデルで動作を確認します。APIへの送信・課金はありません。':'実APIモード：設定したモデルに共通資料を渡します。モデルIDとAPIキーの環境変数を確認してください。';}catch{$('mode-note').textContent='モデル設定のJSONを入力してください。';}
}
function setBusy(value){running=value;for(const id of ['run','validate','load-demo','load-live','open-history'])$(id).disabled=value;for(const id of ['reports','gold','config']){$(id).disabled=value;$('file-'+id).disabled=value;}$('cancel').hidden=!value;if(value)$('cancel').disabled=false;}
function validation(data){
  const box=$('validation');box.replaceChildren();box.hidden=false;
  box.append(el('strong',`${data.reports} 文書 / ${data.items} 項目 / ${data.models} モデル / ${data.repeats} 回`));
  box.append(el('div',`呼び出し上限 ${data.total_calls_max} 回（穴埋め ${data.fill_calls} 回、チェック最大 ${data.check_calls_max} 回）。候補の重複により実際の回数は減ります。`));
  if(!data.demo)box.append(el('div','接続先：'+data.destinations.map(m=>`${m.id} · ${m.model} (${m.base_url||m.provider})`).join(' / ')));
}
async function validate(){clearError();try{validation(await api('/api/validate',inputs()));}catch(err){showError(err.message);}}
async function run(){
  clearError();try{const data=inputs();setBusy(true);validation(await api('/api/validate',data));const job=await api('/api/run',data);jobId=job.id;currentRunId=job.id;$('progress-wrap').hidden=false;$('progress').value=0;poll();}catch(err){setBusy(false);showError(err.message);}
}
async function poll(){
  try{const job=await api('/api/jobs/'+jobId);pollFailures=0;const p=job.progress||{};$('stage').textContent=({prepare:'準備',fill:'穴埋め',check:'ファクトチェック',score:'採点',completed:'完了',done:'完了'})[p.stage]||p.stage||'実行中';$('progress').max=p.total||1;$('progress').value=p.completed||0;$('progress-text').textContent=`${p.completed||0} / ${p.total||0}`;$('progress-message').textContent=p.message||'';
    if(job.status==='running'){setTimeout(poll,750);return;}setBusy(false);$('stage').textContent=labels[job.status]||job.status;if(job.result)renderResults(job.result,job.id);if(job.error)showError(job.error);await refreshHistory();
  }catch(err){pollFailures++;showError('進捗の取得に失敗しました。実験は停止せず、接続を再試行します。 '+err.message);if(pollFailures<=10){setTimeout(poll,Math.min(5000,1000*pollFailures));}else{setBusy(false);showError('進捗を取得できません。ページを再読み込みすると、実行中の実験に再接続できます。 '+err.message);}}
}
function table(target,rows,columns){
  const container=$(target);container.replaceChildren();if(!rows?.length){container.append(el('p','この結果はまだありません。','hint'));return;}
  const t=el('table'),head=el('thead'),tr=el('tr');for(const c of columns)tr.append(el('th',c.label));head.append(tr);t.append(head);const body=el('tbody');for(const row of rows){const line=el('tr');for(const c of columns){const value=typeof c.key==='function'?c.key(row):row[c.key];line.append(el('td',c.format?c.format(value):value??'—'));}body.append(line);}t.append(body);container.append(t);
}
function fillChart(rows){
  const host=$('fill-chart');host.replaceChildren();const legend=el('div',undefined,'chart-legend');for(const [key,label] of [['correct','正答'],['incorrect','誤答'],['abstain','保留'],['failure','実行失敗'],['notrun','未実行']]){const item=el('span');item.append(el('i',undefined,'legend-dot '+key),document.createTextNode(label));legend.append(item);}host.append(legend);
  for(const r of rows||[]){const row=el('div',undefined,'bar-row');row.append(el('span',r.model_id));const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 1000 20');svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('role','img');svg.setAttribute('aria-label',`${r.model_id}: 正答率 ${pct(r.correct_rate)}`);let x=0;for(const [key,cls] of [['correct_rate','correct'],['incorrect_rate','incorrect'],['abstain_rate','abstain'],['error_rate','failure'],['not_run_rate','notrun']]){const v=Math.max(0,Math.min(1,r[key]||0));const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.setAttribute('x',x);rect.setAttribute('y','0');rect.setAttribute('width',v*1000);rect.setAttribute('height','20');rect.setAttribute('class',cls);svg.append(rect);x+=v*1000;}row.append(svg,el('strong',pct(r.correct_rate)));host.append(row);}
}
function renderResults(result,runId){
  currentResult=result;currentRunId=runId;$('empty').hidden=true;$('results').hidden=false;$('download').href='/api/runs/'+runId+'/download';$('result-meta').textContent=`${runId} · ${labels[result.status]||result.status} · ${displayDate(result.created_at)}`;
  $('result-title').textContent=result.run_mode==='recorded_agent_pilot'?'同一LLM・プロンプト条件別の比較':'モデル別の比較';
  const banner=$('result-banner');banner.className='notice'+(result.status!=='completed'?' partial':result.demo?'':' live');banner.textContent=(result.demo?'模擬モデルの結果です。実際のAIの性能を示すものではありません。':'実APIによる実験結果です。')+(result.status!=='completed'?' 途中までの結果です。未実行・失敗件数を含めて確認してください。':'');
  if(result.run_mode==='recorded_agent_pilot')banner.textContent=(result.review?.headline_notice?result.review.headline_notice+' ':'')+'同一の実行環境のLLMが出した回答を取り込み、採点しています。baseline／evidence_firstはプロンプト条件の違いで、別モデルではありません。API費用・速度は未計測です。';
  renderReview(result.review);
  $('warnings').replaceChildren(...(result.warnings||[]).map(w=>el('p',w,'warning-text')));
  fillChart(result.fill_metrics);
  table('fill-table',result.fill_metrics,[{key:'model_id',label:'モデル／条件'},{key:'n',label:'全出題数',format:num},{key:'correct_rate',label:'正答率',format:pct},{key:'incorrect_rate',label:'誤答率',format:pct},{key:'abstain_rate',label:'保留率',format:pct},{key:'error_rate',label:'失敗率',format:pct},{key:'not_run_rate',label:'未実行率',format:pct},{key:'evidence_valid_rate',label:'引用文字列一致率',format:pct},{key:'latency_seconds',label:'合計秒',format:num},{key:'cost',label:'推定費用 USD',format:money}]);
  table('check-table',result.check_metrics,[{key:'model_id',label:'モデル／条件'},{key:'n',label:'全判定数',format:num},{key:'n_supported',label:'正しい候補',format:num},{key:'n_refuted',label:'誤った候補',format:num},{key:'error_detection_rate',label:'誤り検出率',format:pct},{key:'false_flag_rate',label:'正答の誤指摘率',format:pct},{key:'abstain_rate',label:'保留率',format:pct},{key:'error_rate',label:'失敗率',format:pct},{key:'not_run_rate',label:'未実行率',format:pct},{key:'evidence_valid_rate',label:'引用文字列一致率',format:pct},{key:'passed_error_rate',label:'通過結果の誤り率',format:pct},{key:'pass_rate',label:'通過率',format:pct},{key:'cost',label:'推定費用 USD',format:money}]);
  const baseline=(result.fill_metrics||[]).map(r=>({generator_id:r.model_id,checker_id:'チェックなし',n:r.n,correct_provided_rate:r.correct_rate,incorrect_provided_rate:r.incorrect_rate,not_provided_rate:(r.abstain_rate||0)+(r.error_rate||0),not_run_rate:r.not_run_rate||0}));
  table('pipeline-table',[...baseline,...(result.pipeline_metrics||[])],[{key:'generator_id',label:'穴埋めモデル'},{key:'checker_id',label:'チェックモデル'},{key:'n',label:'元の全出題数',format:num},{key:'correct_provided_rate',label:'正しい回答の提供率',format:pct},{key:'incorrect_provided_rate',label:'誤った回答の提供率',format:pct},{key:'not_provided_rate',label:'未提供率',format:pct},{key:'not_run_rate',label:'未実行率',format:pct}]);renderDetails();
}
function renderReview(review){
  $('review-tab').hidden=!review;const host=$('review-content');host.replaceChildren();
  if(!review){if($('review-tab').classList.contains('active'))document.querySelector('[data-tab="fill"]').click();return;}
  host.append(el('h3',review.title||'予備実験のレビュー'),el('p',review.conclusion));
  for(const section of review.sections||[]){host.append(el('h3',section.title));for(const paragraph of section.paragraphs||[])host.append(el('p',paragraph));if(section.bullets?.length){const list=el('ul');for(const text of section.bullets)list.append(el('li',text));host.append(list);}}
  if(review.sources?.length){host.append(el('h3','資料・参照'));const list=el('ul');for(const source of review.sources){const entry=el('li'),link=el('a',source.title);try{const url=new URL(source.url);if(url.protocol==='https:'){link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';}}catch{}entry.append(link);list.append(entry);}host.append(list);}
}
function renderDetails(){
  const host=$('detail-list');host.replaceChildren();if(!currentResult)return;const filter=$('filter').value.toLowerCase();const candidates=new Map((currentResult.candidates||[]).map(c=>[c.id,c]));const rows=[...(currentResult.fills||[]).map(r=>({type:'穴埋め',row:r})),...(currentResult.checks||[]).map(r=>({type:'チェック',row:{...r,candidate:candidates.get(r.candidate_id)||null}})),...(currentResult.errors||[]).map(r=>({type:'エラー',row:r}))];let count=0;
  for(const {type,row} of rows){const serialized=JSON.stringify(row,null,2);if(filter&&!serialized.toLowerCase().includes(filter))continue;count++;if(count>200)continue;const detail=el('details',undefined,'detail-row');const outcome=row.score?.outcome||row.outcome||row.answer?.verdict||row.verdict||row.answer?.status||row.status||'';detail.append(el('summary',`${type} · ${row.item_id||row.candidate_id||''} · ${row.model_id||''} · ${labels[outcome]||outcome}`));detail.append(el('pre',serialized));host.append(detail);}
  host.prepend(el('p',`${count} 件${count>200?'（先頭200件を表示。絞り込みまたはZIPのログで全件を確認できます）':''}`,'filter-note'));
}
async function refreshHistory(){const runs=await api('/api/runs');const select=$('history');select.replaceChildren(el('option','実験を選択'));select.firstChild.value='';for(const run of runs){const option=el('option',`${run.id} · ${runMode(run)} · ${labels[run.status]||run.status}`);option.value=run.id;select.append(option);}if(currentRunId)select.value=currentRunId;}
for(const id of ['reports','gold','config']){$('file-'+id).addEventListener('change',async event=>{clearError();const file=event.target.files[0];if(!file)return;try{setInput(id,JSON.parse((await file.text()).replace(/^\uFEFF/,'')),file.name);$('validation').hidden=true;updateMode();}catch(err){showError(`${file.name}: ${err.message}`);}});$(id).addEventListener('input',()=>{$('validation').hidden=true;updateMode();});}
$('load-demo').addEventListener('click',()=>{for(const id of ['reports','gold','config'])setInput(id,samples[id],'同梱サンプル（架空）');clearError();$('validation').hidden=true;updateMode();});
$('load-live').addEventListener('click',()=>{setInput('config',samples.live_config,'API設定ひな形（モデルIDを編集してください）');$('config').closest('details').open=true;updateMode();$('validation').hidden=true;});
$('validate').addEventListener('click',validate);$('run').addEventListener('click',run);
$('cancel').addEventListener('click',async()=>{try{const res=await api('/api/jobs/'+jobId+'/cancel',{});$('progress-message').textContent=res.message;$('cancel').disabled=true;}catch(err){showError(err.message);}});
$('open-history').addEventListener('click',async()=>{const id=$('history').value;if(!id)return;clearError();try{renderResults(await api('/api/runs/'+id),id);}catch(err){showError(err.message);}});
$('filter').addEventListener('input',renderDetails);
for(const tab of document.querySelectorAll('.tab'))tab.addEventListener('click',()=>{for(const t of document.querySelectorAll('.tab')){const selected=t===tab;t.classList.toggle('active',selected);t.setAttribute('aria-selected',String(selected));$('view-'+t.dataset.tab).hidden=!selected;}});
(async()=>{try{const data=await api('/api/bootstrap');token=data.token;samples=data.samples;for(const id of ['reports','gold','config'])setInput(id,samples[id],'同梱サンプル（架空）');updateMode();await refreshHistory();const requestedRun=new URLSearchParams(location.search).get('run');if(requestedRun&&/^[a-zA-Z0-9_-]+$/.test(requestedRun)){renderResults(await api('/api/runs/'+requestedRun),requestedRun);$('history').value=requestedRun;$('results').scrollIntoView({block:'start'});}if(data.active_job){jobId=data.active_job.id;currentRunId=jobId;setBusy(true);$('progress-wrap').hidden=false;$('mode-note').textContent='実行中の実験に再接続しました。下の入力欄はサンプル表示であり、実行中のデータは変更されません。';poll();}}catch(err){showError('初期データを読み込めませんでした: '+err.message);}})();
