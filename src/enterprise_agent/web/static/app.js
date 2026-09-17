"use strict";
const $ = id => document.getElementById(id);
let lastResult = null;
let askController = null;
const outcomeNames = {supported:"证据支持",partially_supported:"部分支持 / 需核验",unsupported:"证据不足",conflicting:"证据冲突"};
function node(tag, text, cls) { const element=document.createElement(tag); if(text!==undefined) element.textContent=text; if(cls) element.className=cls; return element; }
function safeLink(url) { try { const u=new URL(url); return ["http:","https:"].includes(u.protocol) && !u.username && !u.password ? u.href : null; } catch { return null; } }
function render(result) {
  lastResult=result;
  $("answer").textContent=result.answer;
  $("outcome").textContent=["demo","knowledge-local"].includes(result.mode) ? "本地摘录 · 未做模型审核" : result.mode==="knowledge-model"?"模型回答 · 请核对原文":(outcomeNames[result.outcome] || result.status);
  $("stats").textContent=`${result.agent_elapsed_ms} ms · ${result.tool_call_count} 次工具调用 · ${result.model_token_count??"未知"} model tokens`;
  if(result.mode==="knowledge-agent"){
    $("stats").textContent+=` · ${result.embedding_calls} 次查询向量 / ${result.rerank_calls} 次重排（另行计费）`;
    $("outcome").textContent=`模型审核：${outcomeNames[result.outcome]||result.outcome} · 需核对`;
    $("answer").append(node("p",result.caveat,"knowledge-hint"));
  }
  $("request-id").textContent=`REQUEST ${result.request_id}`;
  $("evidence-count").textContent=`${result.evidence.length} 条`;
  $("evidence").replaceChildren();
  for(const item of result.evidence) {
    const card=node("article",undefined,"evidence-card"); card.id=`evidence-${item.id}`;
    const meta=node("div",undefined,"evidence-meta"); meta.append(node("span",item.id,"evidence-id"),node("span",item.source.toUpperCase()));
    card.append(meta,node("h4",item.title),node("p",item.excerpt));
    if(item.location)card.append(node("small",`来源位置：${item.location} · 切块 ${item.ordinal}`));
    if(result.library_id&&window.knowledge){const view=node("button","查看文档对应原文","secondary");view.type="button";view.onclick=()=>window.knowledge.preview(result.library_id,item.document_id,item.ordinal).catch(error=>{$("error").textContent=error.message;$("error").hidden=false;});card.append(view);}
    const url=safeLink(item.url); if(url){const link=node("a","打开原始来源 ↗");link.href=url;link.target="_blank";link.rel="noopener noreferrer";card.append(link);}
    $("evidence").append(card);
  }
  if(!result.evidence.length) $("evidence").append(node("p","没有可展示的授权证据。"));
  $("facts").replaceChildren();
  for(const item of result.facts) {const row=node("div",undefined,"fact");row.append(node("span",{supported:"已支持",missing:"缺失",conflicting:"冲突"}[item.status]||item.status,"status-tag"),node("strong",item.fact),node("p",item.conclusion));if(item.evidence_ids.length)row.append(node("small",item.evidence_ids.join(" · ")));$("facts").append(row);}
  if(!result.facts.length) $("facts").append(node("p","此请求没有事实审核结果。"));
  $("activity").replaceChildren();
  for(const item of result.activity) {const row=node("div",undefined,"activity-item");row.append(node("strong",`${item.label} · ${item.status}`),node("p",item.detail));$("activity").append(row);}
  $("results").hidden=false;
}
$("ask-form").addEventListener("submit",async event=>{
  event.preventDefault(); const question=$("question").value.trim(); if(!question)return;
  const libraryId=window.knowledge?.selected;
  if(window.enterpriseMode==="workspace"&&!libraryId){$("error").textContent="请先创建并选择知识库，上传文件后再提问。";$("error").hidden=false;return;}
  askController?.abort();const controller=new AbortController();askController=controller;
  $("error").hidden=true;$("results").hidden=true;$("pending").hidden=false;$("submit").disabled=true;lastResult=null;
  const headers={"Content-Type":"application/json"}; const token=$("token").value.trim(); if(token)headers.Authorization=`Bearer ${token}`;
  const endpoint=libraryId?`/api/knowledge/libraries/${libraryId}/ask`:"/ask";
  const answerMode=$("knowledge-model").checked?$("knowledge-answer-kind").value:"local";
  const payload=libraryId?{question,answer_mode:answerMode,retrieval_strategy:answerMode==="agent"?$("knowledge-strategy").value:"bm25"}:{question};
  try {const response=await fetch(endpoint,{method:"POST",headers,signal:controller.signal,body:JSON.stringify(payload)});const result=await response.json();if(!response.ok){const detail=result.detail;throw new Error(response.status===401?"访问令牌无效，请联系管理员。":response.status===429?"服务器正在处理其他调查，请稍后重试。":typeof detail==="string"?detail:detail?.message?`${detail.message}（请求 ${detail.request_id}）`:`请求失败（HTTP ${response.status}）`);}if(libraryId===window.knowledge?.selected)render(result);}
  catch(error){$("error").textContent=error.name==="AbortError"?"已停止等待。已经提交的模型请求可能仍在服务端运行。":error.message;$("error").hidden=false;}
  finally{if(askController===controller){$("pending").hidden=true;$("submit").disabled=false;askController=null;}}
});
$("cancel-ask").addEventListener("click",()=>askController?.abort());
$("export").addEventListener("click",()=>{if(!lastResult)return;const blob=new Blob([JSON.stringify(lastResult,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const link=node("a");link.href=url;link.download=`investigation-${lastResult.request_id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
(async()=>{try{const response=await fetch("/api/config");if(!response.ok)throw new Error("配置读取失败");const config=await response.json();window.enterpriseConfig=config;window.enterpriseMode=config.mode;$("mode").textContent=config.mode==="demo"?"● OFFLINE DEMO":config.mode==="workspace"?"● PERSONAL WORKSPACE":"● LIVE · AUTH REQUIRED";$("mode-notice").textContent=config.mode==="demo"?"内置演示使用 9 篇合成文档，不调用模型。你也可以在“我的知识库”上传自己的资料，并在下方选择检索范围。":config.mode==="workspace"?"个人知识库模式：先连接身份、上传文件，再提问。默认只在本地检索；启用模型回答会发送问题和相关原文给配置的模型。":"企业真实模式：使用配置的模型与索引。问题与证据可能发送给模型供应商；请先确认数据处理规则。个人上传资料与企业索引分开。";$("credentials").hidden=!config.requires_token;for(const question of config.examples){const button=node("button",question);button.type="button";button.addEventListener("click",()=>{$("question").value=question;$("question").focus();});$("examples").append(button);}if(window.knowledge)await window.knowledge.init(config);}catch(error){$("mode").textContent="连接失败";$("mode-notice").textContent="无法连接服务，请检查启动日志后刷新页面。";$("submit").disabled=true;}})();
