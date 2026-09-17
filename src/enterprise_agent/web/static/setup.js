"use strict";
(() => {
  let configuration=null, indexPoll=null, setupBusy=false, indexGeneration=0;
  const api=(path,options)=>window.knowledge.api(path,options);
  const json=body=>({headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  function message(value){$("provider-status").textContent=value;}
  function fill(){
    if(!configuration)return;
    const kind=$("provider-kind").value, value=configuration[kind];
    $("provider-url").value=value.base_url||"";$("provider-model").value=value.model||"";$("provider-key").value="";
    $("provider-dimensions").value=value.dimensions||1024;$("dimension-field").hidden=kind!=="embedding";
    $("provider-test").disabled=!value.configured||setupBusy;$("provider-remove").disabled=!value.configured||setupBusy;
    message(value.configured?`已配置 ${value.model}；密钥${value.key_configured?"已保存（不回显）":"为空"}。` : "尚未配置；本地原文检索不受影响。");
  }
  async function session(){
    const s=window.knowledgeSession;if(!s)return;
    $("setup-panel").hidden=!s.setup_enabled;
    $("index-build").disabled=!s.providers.embedding.configured;
    if(s.setup_enabled&&!setupBusy){try{configuration=await api("/setup");fill();}catch(e){message(e.message);}}
  }
  function choices(){
    $("agent-choices").hidden=!$("knowledge-model").checked;
    $("knowledge-strategy").disabled=$("knowledge-answer-kind").value!=="agent";
  }
  async function storage(){
    const identityToken=$("token").value;
    try{const s=await api("/storage");if(identityToken!==$("token").value)return;
      const mib=bytes=>(bytes/1024/1024).toFixed(2);
      $("storage-usage").textContent=`当前身份：${s.documents} 个文件 · 原文件 ${mib(s.raw_bytes)} / ${mib(s.raw_limit)} MiB · 向量 ${mib(s.vector_bytes)} MiB + 预留 ${mib(s.vector_reserved_bytes)} / ${mib(s.vector_limit)} MiB。${s.note}`;
    }catch(e){$("storage-usage").textContent=e.message;}
  }
  async function index(){
    clearTimeout(indexPoll);const generation=++indexGeneration,id=window.knowledge.selected;if(!id)return;
    try{
      const result=await api(`/libraries/${id}/index`);if(id!==window.knowledge.selected||generation!==indexGeneration)return;
      const names={not_built:"未构建（仍可用关键词检索）",queued:"排队中",building:"构建中",ready:"语义索引就绪",stale:"索引已过期：文档或向量模型变化，请重建",failed:"构建未完成"};
      $("index-status").textContent=`${names[result.status]||result.status} · ${result.done}/${result.total} 个切块 · 本次复用 ${result.reused} / 新生成 ${result.embedded}${result.error?" · "+result.error:""}`;
      $("index-cancel").disabled=!["queued","building","stale"].includes(result.status);
      if(["queued","building"].includes(result.status))indexPoll=setTimeout(index,1500);
      await storage();
    }catch(e){$("index-status").textContent=e.message;}
  }
  async function setupAction(action){
    if(setupBusy)return;setupBusy=true;$("provider-save").disabled=true;$("provider-test").disabled=true;$("provider-remove").disabled=true;
    try{await action();}catch(e){message(e.message);}finally{setupBusy=false;$("provider-save").disabled=false;$("provider-test").disabled=false;$("provider-remove").disabled=false;$("provider-key").value="";}
  }
  $("provider-kind").onchange=fill;
  $("provider-form").onsubmit=e=>{e.preventDefault();setupAction(async()=>{
    const kind=$("provider-kind").value;
    const provider={base_url:$("provider-url").value.trim(),model:$("provider-model").value.trim(),api_key:$("provider-key").value,dimensions:Number($("provider-dimensions").value)};
    configuration=await api(`/setup/${kind}`,{method:"PUT",...json({provider,revision:configuration.revision})});
    fill();await window.knowledge.connect();await index();message("配置已保存，尚未发送测试请求。下一步点击“测试已保存配置”。");
  });};
  $("provider-test").onclick=()=>{if(!confirm("将向已保存的服务发送固定测试文本（不包含您的文件），可能产生费用。继续？"))return;setupAction(async()=>{message("测试中…");const result=await api(`/setup/${$("provider-kind").value}/test`,{method:"POST",...json({consent:true})});message(`连接成功 · ${result.elapsed_ms} ms。${result.note}`);});};
  $("provider-remove").onclick=()=>{if(!confirm("移除此用途的模型配置？不会删除您的文件。"))return;setupAction(async()=>{configuration=await api(`/setup/${$("provider-kind").value}`,{method:"PUT",...json({provider:null,revision:configuration.revision})});fill();$("knowledge-model").checked=false;choices();await window.knowledge.connect();await index();message("已移除配置；历史向量不再用于当前检索。文件仍保留。");});};
  $("index-build").onclick=async()=>{const id=window.knowledge.selected;if(!id)return;$("index-build").disabled=true;
    try{
      const force=$("index-force").checked,plan=await api(`/libraries/${id}/index/plan?force=${force}`);
      if(id!==window.knowledge.selected)return;
      const detail=`共 ${plan.total} 块，可复用 ${plan.reused} 块；本次需发送 ${plan.pending} 块、${plan.characters_to_send} 个字符，预计 ${plan.estimated_batches} 批请求。`;
      $("index-plan").textContent=detail+"字符数不是 token 数或价格估算。";
      if(!confirm(detail+"\n需要发送的内容包含文件名和原文；可能计费。失败后只保证复用已保存批次，不保证供应商未对超时请求计费。确认继续？"))return;
      await api(`/libraries/${id}/index`,{method:"POST",...json({consent:true,plan_id:plan.plan_id,force})});await index();
    }catch(e){$("index-status").textContent=e.message;}finally{$("index-build").disabled=!window.knowledgeSession?.providers.embedding.configured;}
  };
  $("index-cancel").onclick=async()=>{const id=window.knowledge.selected;if(!id)return;try{await api(`/libraries/${id}/index`,{method:"DELETE"});await index();}catch(e){$("index-status").textContent=e.message;}};
  $("index-refresh").onclick=index;
  $("index-clear").onclick=async()=>{const id=window.knowledge.selected;if(!id||!confirm("清除该库的全部语义索引和可复用批次，并取消构建？原文件、关键词检索仍保留。此操作不可撤销，重新向量化可能计费。"))return;try{await api(`/libraries/${id}/vectors`,{method:"DELETE"});$("index-plan").textContent="已清理索引，原文件保留。SQLite 物理文件可能不会立即缩小。";await index();}catch(e){$("index-status").textContent=e.message;}};
  $("knowledge-model").addEventListener("change",choices);$("knowledge-answer-kind").addEventListener("change",choices);
  window.addEventListener("knowledge-selected",()=>{clearTimeout(indexPoll);$("knowledge-strategy").value="bm25";$("index-force").checked=false;$("index-plan").textContent="";choices();index();});
  window.addEventListener("knowledge-session",()=>{session();storage();});
  window.addEventListener("knowledge-documents",()=>{index();storage();});
  $("token").addEventListener("input",()=>{configuration=null;indexGeneration++;clearTimeout(indexPoll);$("setup-panel").hidden=true;$("storage-usage").textContent="";});
  session();
  if(window.knowledgeSession)storage();
})();
