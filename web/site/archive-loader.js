'use strict';
// Classic scripts can load public archive data in an opaque-origin Pages sandbox.
// Payloads are generated from the adjacent JSON files; no external service is used.
window.MASArchive=(()=>{
 const pending=new Map();
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
 function load(path){
  if(!/^data\/[a-zA-Z0-9_/-]+\.json$/.test(path)||path.includes('..'))return Promise.reject(Error('Invalid archive path'));
  if(pending.has(path))return pending.get(path).promise;
  let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});
  const script=document.createElement('script');
  const timeout=setTimeout(()=>fail(),45000);
  const cleanup=()=>{clearTimeout(timeout);script.remove()};
  function fail(){cleanup();pending.delete(path);reject(Error(`Cannot load archived evidence: ${path}`))}
  pending.set(path,{promise,resolve,reject,cleanup});
  script.src=path.replace(/^data\//,'archive/').replace(/\.json$/,'.js');
  script.onerror=fail;document.head.append(script);return promise;
 }
 return {load,receive};
})();
