/* Headless browser smoke test. Starts/stops only its own loopback demo server. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '..');
const python = process.env.EA_PYTHON || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const port = process.env.EA_BROWSER_PORT || '18019';
const origin = `http://127.0.0.1:${port}`;
const output = path.resolve(process.env.EA_SCREENSHOT_DIR || path.join(root, 'artifacts', 'ui'));
fs.mkdirSync(output, {recursive:true});
const server = spawn(python, ['-B','-m','enterprise_agent.cli.serve','--mode','demo','--port',port,'--knowledge-dir',path.join(output,'browser-data')], {cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe']});
let logs=''; server.stdout.on('data',x=>logs+=x);server.stderr.on('data',x=>logs+=x);
let browser;
(async()=>{
  for(let i=0;i<100;i++) {
    if(server.exitCode!==null)throw new Error(`Server exited: ${logs}`);
    try {const response=await fetch(`${origin}/api/config`);if(response.ok)break;} catch {}
    if(i===99)throw new Error('Demo startup timed out');
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  browser=await chromium.launch({headless:true,channel:process.env.EA_BROWSER_CHANNEL || undefined});
  const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(origin);
  await page.waitForFunction(()=>document.getElementById('mode').textContent.includes('OFFLINE DEMO'));
  await page.screenshot({path:path.join(output,'workspace-overview.png'),fullPage:true});
  await page.locator('#examples button').first().click();
  await page.locator('#submit').click();
  await page.waitForSelector('#results:not([hidden])');
  assert.ok(await page.locator('.evidence-card').count()>0);
  assert.match(await page.locator('#answer').innerText(),/E1/);
  assert.match(await page.locator('#stats').innerText(),/0 model tokens/);
  await page.screenshot({path:path.join(output,'investigation-desktop.png'),fullPage:true});
  const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();
  assert.match((await downloadPromise).suggestedFilename(),/^investigation-.*\.json$/);
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(output,'investigation-mobile.png'),fullPage:true});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'Mobile horizontal overflow');
  await page.locator('#question').fill('xylophoniumquasar');await page.locator('#submit').click();
  await page.waitForSelector('#results:not([hidden])');
  assert.equal(await page.locator('.evidence-card').count(),0);
  // Untrusted provider/doc text must be displayed literally, never interpreted as HTML.
  const malicious='<img src=x onerror="window.injected=true">';
  await page.route('**/ask',route=>route.fulfill({json:{answer:malicious,mode:'live',outcome:'supported',status:'completed',request_id:'xss-test',agent_elapsed_ms:1,tool_call_count:1,model_token_count:0,evidence:[{id:'E1',source:'jira',title:malicious,excerpt:malicious,url:'javascript:alert(1)'}],facts:[],activity:[]}}));
  await page.locator('#question').fill('unsafe content');await page.locator('#submit').click();
  await page.waitForFunction(text=>document.getElementById('answer').textContent===text,malicious);
  assert.equal(await page.locator('#answer img').count(),0);
  assert.equal(await page.locator('.evidence-card a').count(),0);
  assert.equal(await page.evaluate(()=>window.injected),undefined);
  assert.deepEqual(errors,[]);
  console.log('Browser checks passed: desktop, mobile, evidence, unknown query, export and HTML/URL injection.');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();server.kill();});
