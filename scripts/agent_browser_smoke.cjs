/* Real browser + real graph + loopback-only synthetic model server. No paid API. */
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http');
const {spawn}=require('node:child_process'),{chromium}=require('playwright');
const root=path.resolve(__dirname,'..'),python=process.env.EA_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const port=process.env.EA_AGENT_BROWSER_PORT||'18039',origin=`http://127.0.0.1:${port}`;
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'ea-agent-browser-')),output=path.join(root,'artifacts','ui');fs.mkdirSync(output,{recursive:true});
let externalCalls=0,server,browser,logs='';
const provider=http.createServer(async(req,res)=>{
  const chunks=[];for await(const part of req)chunks.push(part);const body=chunks.length?JSON.parse(Buffer.concat(chunks).toString()):{};
  res.setHeader('Content-Type','application/json');let response;
  if(req.url==='/health')response={ok:true};
  else if(req.url==='/info')response={model_id:'synthetic-model',model_type:{reranker:{}},max_concurrent_requests:20,max_batch_requests:20,max_client_batch_size:20};
  else if(req.url==='/v1/embeddings'){externalCalls++;response={object:'list',model:'synthetic-model',data:body.input.map((text,index)=>({object:'embedding',index,embedding:[1,0]})),usage:{prompt_tokens:10,total_tokens:10}};}
  else if(req.url==='/rerank'){externalCalls++;response=body.texts.map((text,index)=>({index,score:1/(index+1)}));}
  else if(req.url==='/v1/chat/completions'){
    externalCalls++;const system=body.messages[0].content;let text;
    if(system.includes('You are the planner'))text=JSON.stringify({kind:'search_enterprise_knowledge',query:'Orchid 预算',source:'uploaded_file',top_k:1,reason:'Find budget'});
    else if(system.includes('evidence-sufficiency judge'))text=JSON.stringify({decision:'supported',facts:[{fact:'预算',status:'supported',conclusion:'7800 元',evidence_ids:['E1']}],reason:'Synthetic evidence supports it'});
    else if(system.includes('Connection test'))text='OK';
    else text='Orchid 预算为 7800 元。[E1]';
    response={id:'synthetic',object:'chat.completion',created:0,model:'synthetic-model',choices:[{index:0,message:{role:'assistant',content:text},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:10,total_tokens:20}};
  }else{res.statusCode=404;response={error:'unknown stub route'};}
  res.end(JSON.stringify(response));
});
async function ready(){for(let i=0;i<100;i++){if(server.exitCode!==null)throw new Error(logs);try{if((await fetch(origin+'/api/config')).ok)return;}catch{}await new Promise(r=>setTimeout(r,100));}throw new Error('startup timeout');}
(async()=>{
  await new Promise(resolve=>provider.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${provider.address().port}`;
  server=spawn(python,['-B','-m','enterprise_agent.cli.serve','--mode','demo','--port',port,'--enable-setup','--knowledge-dir',temporary],{cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe'],env:{...process.env,AGENT_KNOWLEDGE_MODEL_ENABLED:'0'}});
  server.stdout.on('data',x=>logs+=x);server.stderr.on('data',x=>logs+=x);await ready();
  browser=await chromium.launch({headless:true,channel:process.env.EA_BROWSER_CHANNEL||undefined});const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
  await page.goto(origin);await page.waitForSelector('#setup-panel:not([hidden])');await page.locator('#setup-panel summary').click();
  async function configure(kind){
    await page.locator('#provider-kind').selectOption(kind);await page.locator('#provider-url').fill(base+(kind==='reranker'?'':'/v1'));await page.locator('#provider-model').fill('synthetic-model');
    await page.locator('#provider-key').fill('browser-test-secret');if(kind==='embedding')await page.locator('#provider-dimensions').fill('2');
    const before=externalCalls;await page.locator('#provider-save').click();await page.waitForFunction(()=>document.getElementById('provider-status').textContent.includes('配置已保存'));
    assert.equal(externalCalls,before,'save must not contact provider');assert.equal(await page.locator('#provider-key').inputValue(),'');
    await page.locator('#provider-test').click();await page.waitForFunction(()=>document.getElementById('provider-status').textContent.includes('连接成功'));assert.ok(externalCalls>before);
  }
  await configure('llm');await configure('embedding');await configure('reranker');
  const summary=await page.evaluate(async()=>await (await fetch('/api/knowledge/setup')).text());assert.ok(!summary.includes('browser-test-secret'));
  await page.locator('#knowledge-nav').click();await page.locator('#library-name').fill('完整 Agent 测试');await page.locator('#library-create').click();await page.waitForSelector('#document-workspace:not([hidden])');
  const beforeUpload=externalCalls;
  await page.locator('#file-upload').setInputFiles({name:'Orchid.txt',mimeType:'text/plain',buffer:Buffer.from('Orchid 项目预算为 7800 元。负责人是 Ada。')});
  await page.waitForFunction(()=>document.getElementById('document-list').textContent.includes('可用'),{},{timeout:20000});assert.equal(externalCalls,beforeUpload,'upload must not call provider');
  await page.locator('#index-build').click();await page.waitForFunction(()=>document.getElementById('index-status').textContent.includes('语义索引就绪'),{},{timeout:20000});
  const afterFirstIndex=externalCalls;
  await page.locator('#index-build').click();await page.waitForFunction(()=>document.getElementById('index-status').textContent.includes('复用 1 / 新生成 0'),{},{timeout:20000});
  assert.equal(externalCalls,afterFirstIndex,'unchanged rebuild must reuse without a provider call');
  assert.match(await page.locator('#index-plan').innerText(),/本次需发送 0 块/);
  await page.locator('#file-upload').setInputFiles({name:'补充资料.txt',mimeType:'text/plain',buffer:Buffer.from('备用项目由 Lin 负责。')});
  await page.waitForFunction(()=>{const rows=[...document.querySelectorAll('.document-row')];return rows.length===2&&rows.every(r=>r.textContent.includes('可用'));},{},{timeout:20000});
  await page.locator('#index-build').click();await page.waitForFunction(()=>document.getElementById('index-status').textContent.includes('复用 1 / 新生成 1'),{},{timeout:20000});
  assert.equal(externalCalls,afterFirstIndex+1,'only the added file should be embedded');
  assert.match(await page.locator('#index-plan').innerText(),/本次需发送 1 块/);
  await page.locator('#knowledge-model').check();await page.locator('#knowledge-strategy').selectOption('reranked');await page.locator('#question').fill('Orchid 预算是多少？');await page.locator('#submit').click();
  await page.waitForSelector('#results:not([hidden])',{timeout:30000});assert.match(await page.locator('#answer').innerText(),/7800/);assert.match(await page.locator('#activity').innerText(),/evidence_judge/);
  assert.match(await page.locator('#outcome').innerText(),/证据支持/);assert.equal(await page.locator('.evidence-card').count(),1);
  const result=await page.evaluate(()=>lastResult);assert.equal(result.mode,'knowledge-agent');assert.equal(result.embedding_calls,1);assert.equal(result.rerank_calls,1);assert.equal(result.model_call_count,3);
  await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(output,'agent-setup-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'mobile overflow');await page.screenshot({path:path.join(output,'agent-setup-mobile.png'),fullPage:true});
  await page.locator('#index-clear').click();await page.waitForFunction(()=>document.getElementById('index-status').textContent.includes('未构建'));
  assert.equal(await page.locator('.document-row').count(),2,'clear vectors must preserve documents');
  const beforeBlocked=externalCalls;
  const blocked=await page.evaluate(async()=>{const response=await fetch(`/api/knowledge/libraries/${window.knowledge.selected}/ask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:'Orchid',answer_mode:'agent',retrieval_strategy:'hybrid'})});return response.status;});
  assert.equal(blocked,409);assert.equal(externalCalls,beforeBlocked,'missing index must fail before model calls');
  await page.reload();await page.waitForFunction(()=>window.knowledgeSession?.model_enabled);assert.equal(await page.locator('#knowledge-model').isChecked(),false);
  assert.deepEqual(errors,[]);console.log('Agent browser passed: setup, masked keys, explicit plan/consent, zero-call reuse, incremental file indexing, hybrid + rerank + graph, citations, clear-index without file deletion, mobile and opt-in reset. Synthetic loopback providers only.');
})().catch(e=>{console.error(e);console.error(logs);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(server)server.kill();provider.close();console.log('Disposable test data: '+temporary);});
