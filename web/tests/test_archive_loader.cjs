const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),z=require('node:zlib');
const root=require('node:path').join(__dirname,'../site/');
global.window=global;let appended=0;
global.document={createElement(){return {remove(){}}},head:{append(s){appended++;setImmediate(()=>{const key=s.src.replace(/^archive\//,'data/').replace(/\.js$/,'.json');if(key.includes('missing'))return s.onerror();MASArchive.receive(key,z.gzipSync(Buffer.from(JSON.stringify({key,value:'<script> literal'}))).toString('base64'));})}}};
vm.runInThisContext(fs.readFileSync(root+'archive-loader.js','utf8'));
(async()=>{const [a,b]=await Promise.all([MASArchive.load('data/test.json'),MASArchive.load('data/test.json')]);assert.equal(appended,1);assert.equal(a.value,'<script> literal');assert.deepEqual(a,b);await assert.rejects(MASArchive.load('../private.json'));await assert.rejects(MASArchive.load('data/missing.json'));console.log('PASS: sandbox loader, deduplication, exact data, invalid path and load failure');})().catch(e=>{console.error(e);process.exitCode=1});
