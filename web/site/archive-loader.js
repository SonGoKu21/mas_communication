'use strict';
// Classic scripts can load public archive data in an opaque-origin Pages sandbox.
// Payloads are generated from the adjacent JSON files; no external service is used.
window.MASArchive=(()=>{
 const pending=new Map();let bootstrap;
 async function receive(path,payload){
  const entry=pending.get(path);if(!entry)return;
  try{
   const bytes=Uint8Array.from(atob(payload),c=>c.charCodeAt(0));
   const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
   const data=JSON.parse(await new Response(stream).text());
   entry.resolve(data);
  }catch(error){entry.reject(error)}
  finally{entry.cleanup();pending.delete(path)}
 }
 function request(path){
  if(!/^data\/[a-zA-Z0-9_/-]+\.json$/.test(path)||path.includes('..'))return Promise.reject(Error('Invalid archive path'));
  if(pending.has(path))return pending.get(path).promise;
  let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});
  const script=document.createElement('script');
  let timeout,attempt=0;
  const cleanup=()=>{clearTimeout(timeout);script.remove()};
  function fail(){cleanup();if(++attempt<3){setTimeout(start,attempt*2500);return}pending.delete(path);reject(Error(`Archived data temporarily unavailable: ${path}. Please try again shortly.`))}
  function start(){timeout=setTimeout(fail,45000);script.src=path.replace(/^data\//,'archive/').replace(/\.json$/,'.js')+'?v=20261011-bundle&attempt='+attempt;document.head.append(script)}
  pending.set(path,{promise,resolve,reject,cleanup});
  script.onerror=fail;start();return promise;
 }
 function load(path){
  if(/^data\/(manifest|findings|task-map|catalog-\d+)\.json$/.test(path)){
   if(!bootstrap)bootstrap=request('data/bootstrap.json').catch(error=>{bootstrap=null;throw error});
   return bootstrap.then(data=>{if(!(path in data))throw Error('Missing startup data: '+path);return data[path]});
  }
  return request(path);
 }
 return {load,receive};
})();
