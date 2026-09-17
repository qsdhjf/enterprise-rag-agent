"use strict";
(() => {
  let libraries=[], selected="", modelEnabled=false, connected=false, poll=null, uploading=false, previewState=null;
  const statusNames={queued:"排队中",parsing:"正在解析",indexing:"正在建立检索数据",ready:"可用",failed:"失败"};
  function headers(extra={}) {const result={...extra};const token=$("token").value.trim();if(token)result.Authorization=`Bearer ${token}`;return result;}
  function error(message){$("knowledge-error").textContent=message;$("knowledge-error").hidden=!message;}
  async function api(path, options={}) {
    const response=await fetch(`/api/knowledge${path}`,{...options,headers:headers(options.headers)});
    if(response.status===204)return null;
    const result=await response.json();
    if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:`请求失败（HTTP ${response.status}）`);
    return result;
  }
  const libraryPath=id=>`/libraries/${encodeURIComponent(id)}`;
  function clearResult(){lastResult=null;$("results").hidden=true;}
  function select(id){selected=id;$("knowledge-source").value=id;$("knowledge-model").checked=false;$("knowledge-model").disabled=!modelEnabled;$("model-choice").hidden=!id;$("model-help").textContent=modelEnabled?"":"（未配置对话模型；本地管理员可用 --enable-setup 开启上方向导。当前可使用原文检索。）";clearResult();renderLibraries();loadDocuments().catch(e=>error(e.message));window.dispatchEvent(new Event("knowledge-selected"));}
  function renderLibraries(){
    $("library-list").replaceChildren();
    const current=$("knowledge-source").value;
    const options=[node("option",window.enterpriseMode==="workspace"?"请选择知识库":"内置样例 / 企业索引")];options[0].value="";
    for(const library of libraries){
      const option=node("option",library.name);option.value=library.id;options.push(option);
      const button=node("button",library.name);button.type="button";button.setAttribute("aria-pressed",String(library.id===selected));button.onclick=()=>select(library.id);$("library-list").append(button);
    }
    $("knowledge-source").replaceChildren(...options);$("knowledge-source").value=libraries.some(l=>l.id===current)?current:selected;
    $("document-workspace").hidden=!selected;
    $("library-title").textContent=libraries.find(l=>l.id===selected)?.name||"";
    if(!libraries.length)$("library-list").append(node("small","还没有知识库。创建一个，然后上传你自己的文件。"));
  }
  async function connect(){
    error("");
    const connectingToken=$("token").value.trim();
    const session=await api("/session");modelEnabled=session.model_enabled;connected=true;$("knowledge-model").disabled=!modelEnabled;
    window.knowledgeSession=session;window.dispatchEvent(new Event("knowledge-session"));
    if(window.enterpriseMode==="demo"&&Object.values(session.providers).some(p=>p?.configured))$("mode").textContent="● LOCAL · 模型需手动启用";
    const result=await api("/libraries");if(connectingToken!==$("token").value.trim())return;libraries=result;renderLibraries();
    $("knowledge-hint").textContent=window.enterpriseMode==="demo"?
      "个人文件保存在本机，不会进入内置样例。浏览器会话用于区分资料；清除 Cookie 或换浏览器后不能自动访问原库。重要资料建议使用带身份认证的 workspace 模式，并做好备份。":
      "资料绑定当前租户和用户身份，保存在部署服务器；仅自己可访问。默认本地关键词检索，勾选模型回答才发送相关片段给模型。";
    if(window.enterpriseMode==="workspace"&&libraries.length&&!selected)select(libraries[0].id);
  }
  async function loadDocuments(){
    clearTimeout(poll);if(!selected)return;
    const id=selected, documents=await api(`${libraryPath(id)}/documents`);if(id!==selected)return;
    $("document-list").replaceChildren();
    for(const doc of documents){
      const row=node("article",undefined,"document-row");row.dataset.documentId=doc.id;
      const info=node("div",undefined,"document-info");info.append(node("strong",doc.name),node("small",`${statusNames[doc.status]||doc.status} · ${(doc.size/1024).toFixed(1)} KiB · ${doc.chunk_count} 个切块 · 版本 ${doc.revision}`));
      if(doc.error)info.append(node("p",doc.error));for(const warning of doc.warnings)info.append(node("p",warning));
      const actions=node("div",undefined,"document-actions");
      function button(label, action){const b=node("button",label,"secondary");b.type="button";b.onclick=async()=>{b.disabled=true;error("");try{await action();}catch(e){error(e.message);}finally{b.disabled=false;}};actions.append(b);return b;}
      if(doc.status==="ready")button("查看原文",()=>preview(id,doc.id));
      if(doc.status==="failed")button("重试",async()=>{await api(`${libraryPath(id)}/documents/${doc.id}/retry`,{method:"POST"});await loadDocuments();});
      const replacement=node("input");replacement.type="file";replacement.accept=".txt,.md,.pdf,.docx";
      replacement.onchange=()=>{if(replacement.files.length)uploadFiles(replacement.files,doc.id,id);};actions.append(replacement);
      button("替换文件",()=>replacement.click());
      button("删除",async()=>{if(!confirm(`删除“${doc.name}”？文件和切块会从当前知识库移除，不能撤销。`))return;await api(`${libraryPath(id)}/documents/${doc.id}`,{method:"DELETE"});clearResult();await loadDocuments();});
      row.append(info,actions);$("document-list").append(row);
    }
    if(!documents.length)$("document-list").append(node("p","暂无文件。上传完成后系统会自动解析。","knowledge-hint"));
    window.dispatchEvent(new Event("knowledge-documents"));
    if(documents.some(doc=>["queued","parsing","indexing"].includes(doc.status)))poll=setTimeout(()=>loadDocuments().catch(e=>error(e.message)),1200);
  }
  async function uploadFiles(files,replaceId=null,id=selected){
    if(uploading)return;if(!id){error("请先创建或选择知识库");return;}
    if(files.length>20){error("每批最多上传 20 个文件");return;}
    uploading=true;$("file-upload").disabled=true;error("");let succeeded=0;const failures=[];
    try{
      for(const file of files){
        $("upload-status").textContent=`正在上传 ${succeeded+failures.length+1}/${files.length}：${file.name}`;
        if(file.size>8*1024*1024||file.size===0){failures.push(`${file.name}：必须为 1 字节至 8 MiB`);continue;}
        try{await api(`${libraryPath(id)}/documents${replaceId?`/${replaceId}`:""}?filename=${encodeURIComponent(file.name)}`,{method:replaceId?"PUT":"POST",headers:{"Content-Type":"application/octet-stream"},body:file});succeeded++;}
        catch(e){failures.push(`${file.name}：${e.message}`);}
      }
      $("upload-status").textContent=`已接收 ${succeeded} 个文件，后台正在处理。${failures.length?` ${failures.length} 个上传失败。`:""}`;
      if(failures.length)error(failures.join("；"));clearResult();await loadDocuments();
    }finally{uploading=false;$("file-upload").disabled=false;$("file-upload").value="";}
  }
  async function preview(libraryId,documentId,ordinal=1){
    const offset=Math.floor((ordinal-1)/50)*50;
    const result=await api(`${libraryPath(libraryId)}/documents/${documentId}?offset=${offset}`);
    previewState={libraryId,documentId,next:offset+result.chunks.length,total:result.chunk_count};
    $("preview-title").textContent=result.name;$("preview-description").textContent=`版本 ${result.revision} · ${result.chunk_count} 个切块。以下是实际参与检索的提取文本，不是原文件版式。 ${result.warnings.join(" ")}`;
    $("preview-chunks").replaceChildren();appendChunks(result.chunks,ordinal);
    $("preview-more").hidden=previewState.next>=previewState.total;$("document-preview").showModal();
    $("preview-chunks").querySelector(".target")?.scrollIntoView({block:"nearest"});
  }
  function appendChunks(chunks,target){for(const chunk of chunks){const item=node("section",undefined,`preview-chunk${chunk.ordinal===target?" target":""}`);item.append(node("strong",`切块 ${chunk.ordinal} · ${chunk.location}`),node("p",chunk.text));$("preview-chunks").append(item);}}
  $("preview-more").onclick=async()=>{try{const s=previewState;const result=await api(`${libraryPath(s.libraryId)}/documents/${s.documentId}?offset=${s.next}`);appendChunks(result.chunks);s.next+=result.chunks.length;$("preview-more").hidden=s.next>=result.chunk_count;}catch(e){error(e.message);$("document-preview").close();}};
  $("knowledge-nav").onclick=()=>{$("knowledge-panel").open=true;};
  $("knowledge-connect").onclick=()=>connect().catch(e=>error(e.message));
  $("library-create").onclick=async()=>{error("");const name=$("library-name").value.trim();if(!name){error("请填写知识库名称");return;}$("library-create").disabled=true;try{if(!connected)await connect();const item=await api("/libraries",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});libraries=await api("/libraries");$("library-name").value="";select(item.id);}catch(e){error(e.message);}finally{$("library-create").disabled=false;}};
  $("library-delete").onclick=async()=>{if(!selected||!confirm("删除此知识库及其中全部文件？此操作不能撤销。"))return;try{await api(libraryPath(selected),{method:"DELETE"});libraries=await api("/libraries");select("");}catch(e){error(e.message);}};
  $("documents-refresh").onclick=()=>loadDocuments().catch(e=>error(e.message));
  $("file-upload").onchange=()=>uploadFiles(Array.from($("file-upload").files));
  $("knowledge-source").onchange=()=>select($("knowledge-source").value);
  $("token").addEventListener("input",()=>{connected=false;libraries=[];selected="";$("knowledge-model").checked=false;$("model-choice").hidden=true;clearTimeout(poll);renderLibraries();clearResult();});
  const zone=$("drop-zone");for(const event of ["dragenter","dragover"]){zone.addEventListener(event,e=>{e.preventDefault();zone.classList.add("dragging");});}
  for(const event of ["dragleave","drop"]){zone.addEventListener(event,e=>{e.preventDefault();zone.classList.remove("dragging");if(event==="drop")uploadFiles(Array.from(e.dataTransfer.files));});}
  window.knowledge={
    get selected(){return selected;},headers,preview,api,connect,
    init:async config=>{
      window.enterpriseMode=config.mode;renderLibraries();
      if(config.mode==="demo"){
        try{await connect();}catch(e){$("knowledge-hint").textContent=e.message;}
      }else{
        $("knowledge-hint").textContent="填写上方服务令牌，然后点击“连接我的知识库”。";
        if(config.mode==="workspace")$("knowledge-panel").open=true;
      }
    }
  };
  if(window.enterpriseConfig)window.knowledge.init(window.enterpriseConfig);
})();
