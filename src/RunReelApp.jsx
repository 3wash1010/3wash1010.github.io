import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Trash2,
  Film,
  Loader,
  CheckCircle,
  Camera,
  Zap,
  Music2,
  Volume2,
  RotateCw,
  Type,
} from "lucide-react";
import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";
import "@tensorflow/tfjs-backend-webgl";

async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
  ctx.close();
  return audioBuffer;
}

const RunReelApp = () => { 
  const REEL_WIDTH = 720;
  const REEL_HEIGHT = 1280;
  const FPS = 30;
  const VIDEO_SCAN_MAX_SECS = 6;
  const VIDEO_SCAN_FPS = 1;
  const RUNNING_THRESHOLD = 0.4;

  const [files, setFiles] = useState([]);
  const [processedFiles, setProcessedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("upload");
  const [stats, setStats] = useState({
    totalFiles: 0,
    runningDetected: 0,
    nonRunning: 0,
    avgConfidence: 0,
  });

  const [musicFile, setMusicFile] = useState(null);
  const [musicVol, setMusicVol] = useState(0.8);
  const [fadeInSec, setFadeInSec] = useState(0.75);
  const [fadeOutSec, setFadeOutSec] = useState(0.5);
  const [reelUrl, setReelUrl] = useState(null);
  const [numClips, setNumClips] = useState(10);
  const [timePerClip, setTimePerClip] = useState(3);
  const [startCaption, setStartCaption] = useState("");
  const [endCaption, setEndCaption] = useState("");
  const [captionDuration, setCaptionDuration] = useState("dynamic");
  const [fixedCaptionSecs, setFixedCaptionSecs] = useState(5);
  const [imageEffect, setImageEffect] = useState("zoomIn");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatingProgress, setGeneratingProgress] = useState(0);

  const fileInputRef = useRef(null);
  const musicInputRef = useRef(null);
  const detectorRef = useRef(null);
  const renderCanvasRef = useRef(null);
  const previewVideoRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        detectorRef.current = await poseDetection.createDetector(
          poseDetection.SupportedModels.BlazePose,
          { runtime: "tfjs", modelType: "lite" }
        );
        await tf.nextFrame();
        console.log("✅ BlazePose loaded.");
      } catch (err) {
        console.error("❌ Error loading pose model:", err);
      }
    };
    loadModel();
  }, []);

  const runningConfidenceFromPose = (pose, width, height) => {
    if (!pose || !pose.keypoints?.length) return { confidence: 0, center: null };
    const k = (name) => pose.keypoints.find((p) => p.name === name);

    const leftAnkle = k("left_ankle");
    const rightAnkle = k("right_ankle");
    const leftHip = k("left_hip");
    const rightHip = k("right_hip");
    const leftWrist = k("left_wrist");
    const rightWrist = k("right_wrist");
    const nose = k("nose");

    if (!leftAnkle || !rightAnkle || !leftHip || !rightHip) 
      return { confidence: 0, center: null };

    const normX = (p) => p.x / width;
    const normY = (p) => p.y / height;

    const legSpread = Math.abs(normX(leftAnkle) - normX(rightAnkle));
    const hipWidth = Math.abs(normX(leftHip) - normX(rightHip));
    const hipTilt = Math.abs(normY(leftHip) - normY(rightHip));
    let armSwing = 0;
    if (leftWrist?.score > 0.5 && rightWrist?.score > 0.5) {
      armSwing = Math.abs(normX(leftWrist) - normX(rightWrist));
    }

    const strideFactor = legSpread / (hipWidth + 0.01);
    const runningConfidence = Math.min(
      1,
      (strideFactor * 0.6 + armSwing * 0.3 + hipTilt * 0.1) * 2.5
    );

    let centerX = width / 2;
    let centerY = height / 2;
    if (nose && nose.score > 0.3) {
      centerX = nose.x;
      centerY = nose.y;
    } else if (leftHip && rightHip) {
      centerX = (leftHip.x + rightHip.x) / 2;
      centerY = (leftHip.y + rightHip.y) / 2;
    }

    return { 
      confidence: runningConfidence, 
      center: { x: centerX, y: centerY } 
    };
  };

  const waitEvent = (el, event) =>
    new Promise((resolve) => {
      const handler = () => {
        el.removeEventListener(event, handler);
        resolve();
      };
      el.addEventListener(event, handler, { once: true });
    });

  const detectFromFile = async (file) => {
    const detector = detectorRef.current;
    if (!detector) {
      return { 
        type: "unknown", 
        url: "", 
        isRunning: false, 
        confidence: 0,
        timestamp: file.lastModified || Date.now(),
      };
    }
    const blobUrl = URL.createObjectURL(file);
    const timestamp = file.lastModified || Date.now();

    if (file.type.startsWith("image/")) {
      const img = await new Promise((resolve, reject) => {
        const i = new Image();
        i.onload = () => resolve(i);
        i.onerror = reject;
        i.src = blobUrl;
      });

      await new Promise((r) => setTimeout(r, 100));
      const poses = await detector.estimatePoses(img);
      const result = runningConfidenceFromPose(poses?.[0], img.width, img.height);
      const isRunning = result.confidence > RUNNING_THRESHOLD;

      return {
        type: "image",
        url: blobUrl,
        displayUrl: blobUrl,
        isRunning,
        confidence: result.confidence,
        center: result.center,
        width: img.width,
        height: img.height,
        timestamp,
        rotation: 0,
      };
    }

    if (file.type.startsWith("video/")) {
      const video = document.createElement("video");
      video.src = blobUrl;
      video.preload = "auto";
      video.muted = true;
      video.playsInline = true;

      await waitEvent(video, "loadedmetadata");
      if (video.readyState < 2) {
        await Promise.race([waitEvent(video, "canplay"), waitEvent(video, "canplaythrough")]);
      }

      const helper = document.createElement("canvas");
      helper.width = Math.max(2, video.videoWidth || 640);
      helper.height = Math.max(2, video.videoHeight || 360);
      const hctx = helper.getContext("2d");

      const scanDuration = Math.min(VIDEO_SCAN_MAX_SECS, video.duration || VIDEO_SCAN_MAX_SECS);
      const step = 1 / VIDEO_SCAN_FPS;
      let bestConf = 0;
      let detected = false;
      let detectedCenter = null;

      for (let t = 0; t < scanDuration; t += step) {
        video.currentTime = Math.min(t, (video.duration || scanDuration) - 0.01);
        await waitEvent(video, "seeked");
        if (video.readyState < 2) {
          await Promise.race([waitEvent(video, "canplay"), waitEvent(video, "canplaythrough")]);
        }
        hctx.drawImage(video, 0, 0, helper.width, helper.height);
        const poses = await detector.estimatePoses(helper);
        const result = runningConfidenceFromPose(poses?.[0], helper.width, helper.height);
        if (result.confidence > bestConf) {
          bestConf = result.confidence;
          detectedCenter = result.center;
        }
        if (result.confidence > RUNNING_THRESHOLD) {
          detected = true;
          break;
        }
        await new Promise((r) => setTimeout(r, 0));
      }

      hctx.drawImage(video, 0, 0, helper.width, helper.height);
      const thumbUrl = helper.toDataURL("image/png");

      return {
        type: "video",
        url: blobUrl,
        thumbUrl,
        displayUrl: thumbUrl,
        isRunning: detected,
        confidence: bestConf,
        center: detectedCenter,
        startTime: 0,
        duration: video.duration || 1,
        videoWidth: helper.width,
        videoHeight: helper.height,
        timestamp,
        rotation: 0,
      };
    }

    return { 
      type: "unknown", 
      url: blobUrl, 
      isRunning: false, 
      confidence: 0,
      timestamp,
    };
  };

  const handleFileUpload = (e) => {
    const uploaded = Array.from(e.target.files || []);
    setFiles((prev) => [...prev, ...uploaded]);
    setCurrentStep("ready");
  };

  const handleMusicUpload = (e) => {
    const f = e.target.files?.[0];
    setMusicFile(f || null);
  };

  const clearAll = () => {
    setFiles([]);
    setProcessedFiles([]);
    setStats({ totalFiles: 0, runningDetected: 0, nonRunning: 0, avgConfidence: 0 });
    setCurrentStep("upload");
    setReelUrl(null);
    setMusicFile(null);
  };

  const processFiles = async () => {
    if (!detectorRef.current) {
      alert("Pose model is still loading — please wait a moment.");
      return;
    }
    if (files.length === 0) return;

    setIsProcessing(true);
    setProgress(0);
    setCurrentStep("processing");

    const results = [];
    let totalConfidence = 0;

    for (let i = 0; i < files.length; i++) {
      const r = await detectFromFile(files[i]);
      results.push(r);
      if (r.isRunning) totalConfidence += r.confidence;
      setProgress(Math.round(((i + 1) / files.length) * 100));
    }

    const running = results.filter((r) => r.isRunning);
    const nonRunning = results.filter((r) => !r.isRunning);
    running.sort((a, b) => a.timestamp - b.timestamp);

    setStats({
      totalFiles: files.length,
      runningDetected: running.length,
      nonRunning: nonRunning.length,
      avgConfidence: running.length > 0 ? totalConfidence / running.length : 0,
    });

    setProcessedFiles(running);
    setIsProcessing(false);
    setCurrentStep("results");
  };

  const rotateClip = (index) => {
    setProcessedFiles((prev) => {
      const newFiles = [...prev];
      newFiles[index] = {
        ...newFiles[index],
        rotation: ((newFiles[index].rotation || 0) + 90) % 360,
      };
      return newFiles;
    });
  };

  const drawImageFitVertical = (ctx, canvas, img, rotation = 0, center = null, effect = "none", effectProgress = 0) => {
    const TARGET_AR = canvas.width / canvas.height;
    ctx.save();
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let imgW = img.videoWidth || img.width;
    let imgH = img.videoHeight || img.height;
    
    if (rotation === 90 || rotation === 270) {
      [imgW, imgH] = [imgH, imgW];
    }

    const IMG_AR = imgW / imgH;
    let srcX = 0, srcY = 0, srcW = img.width || img.videoWidth, srcH = img.height || img.videoHeight;
    const origW = img.width || img.videoWidth;
    const origH = img.height || img.videoHeight;

    if (rotation === 0 || rotation === 180) {
      if (IMG_AR > TARGET_AR) {
        srcH = origH;
        srcW = srcH * TARGET_AR;
        if (center) {
          srcX = Math.max(0, Math.min(origW - srcW, center.x - srcW / 2));
        } else {
          srcX = (origW - srcW) / 2;
        }
        srcY = 0;
      } else {
        srcW = origW;
        srcH = srcW / TARGET_AR;
        srcX = 0;
        if (center) {
          srcY = Math.max(0, Math.min(origH - srcH, center.y - srcH / 2));
        } else {
          srcY = (origH - srcH) / 2;
        }
      }
    }

    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate((rotation * Math.PI) / 180);

    let scale = 1;
    if (effect === "zoomIn") {
      scale = 1 + effectProgress * 0.15;
    } else if (effect === "zoomOut") {
      scale = 1.15 - effectProgress * 0.15;
    }
    
    if (scale !== 1) {
      ctx.scale(scale, scale);
    }

    const drawW = canvas.width;
    const drawH = canvas.height;
    const drawX = -drawW / 2;
    const drawY = -drawH / 2;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, srcX, srcY, srcW, srcH, drawX, drawY, drawW, drawH);
    ctx.restore();
  };

  const drawCaptionCard = (ctx, canvas, text, bgColor = "#0b0820") => {
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (!text) return;

    ctx.fillStyle = "white";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    
    const maxWidth = canvas.width * 0.85;
    let fontSize = 90;
    ctx.font = `bold ${fontSize}px system-ui, Arial, sans-serif`;
    
    while (ctx.measureText(text).width > maxWidth && fontSize > 40) {
      fontSize -= 5;
      ctx.font = `bold ${fontSize}px system-ui, Arial, sans-serif`;
    }
    
    const words = text.split(' ');
    const lines = [];
    let currentLine = words[0];
    
    for (let i = 1; i < words.length; i++) {
      const testLine = currentLine + ' ' + words[i];
      if (ctx.measureText(testLine).width > maxWidth) {
        lines.push(currentLine);
        currentLine = words[i];
      } else {
        currentLine = testLine;
      }
    }
    lines.push(currentLine);
    
    const lineHeight = fontSize * 1.4;
    const totalHeight = lines.length * lineHeight;
    let y = (canvas.height - totalHeight) / 2 + fontSize / 2;
    
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 20;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 4;
    
    lines.forEach(line => {
      ctx.fillText(line, canvas.width / 2, y);
      y += lineHeight;
    });
    
    ctx.shadowColor = "transparent";
  };

  const drawTitleCard = (ctx, canvas) => {
    ctx.fillStyle = "#0b0820";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.font = "bold 120px system-ui, Arial, sans-serif";
    ctx.textAlign = "center";
    
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 20;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 4;
    
    ctx.fillText("RunReel", canvas.width / 2, canvas.height / 2 - 40);
    ctx.font = "42px system-ui, Arial, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText("Running highlights", canvas.width / 2, canvas.height / 2 + 60);
    ctx.shadowColor = "transparent";
  };

  const onVideoFrame = (video, cb) => {
    if (typeof video.requestVideoFrameCallback === "function") {
      const id = video.requestVideoFrameCallback((...args) =>
        cb(() => video.requestVideoFrameCallback, ...args)
      );
      return () => video.cancelVideoFrameCallback?.(id);
    }
    let rafId = 0;
    const tick = (t) => {
      cb(() => requestAnimationFrame, t, { mediaTime: video.currentTime, presentedFrames: 0 });
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  };

  const getCaptionDuration = (text) => {
    if (captionDuration === "fixed") {
      return fixedCaptionSecs;
    }
    const words = text.trim().split(/\s+/).length;
    return Math.max(2, Math.min(8, words * 0.5));
  };

  const generateReel = async () => {
    if (processedFiles.length === 0) {
      alert("No running clips to compile.");
      return;
    }

    setIsGenerating(true);
    setGeneratingProgress(0);

    const clipsToUse = processedFiles.slice(0, numClips);
    const clipDuration = Math.max(0.5, Math.min(10, timePerClip));

    const canvas = renderCanvasRef.current;
    const ctx = canvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
      willReadFrequently: false
    });

    const videoStream = canvas.captureStream(FPS);
    const videoTrack = videoStream.getVideoTracks()[0];
    
    if (videoTrack && videoTrack.applyConstraints) {
      try {
        await videoTrack.applyConstraints({
          width: { ideal: REEL_WIDTH },
          height: { ideal: REEL_HEIGHT },
          frameRate: { ideal: FPS }
        });
      } catch (e) {
        console.log("Could not apply track constraints:", e);
      }
    }

    const startCaptionDur = startCaption ? getCaptionDuration(startCaption) : 0;
    const endCaptionDur = endCaption ? getCaptionDuration(endCaption) : 0;
    const titleDur = startCaption ? 0 : 0.75;

    const totalReelSecs =
      titleDur +
      startCaptionDur +
      clipsToUse.length * clipDuration +
      endCaptionDur +
      (endCaption ? 0 : 0.5);

        // ---------------------------------------------------------
    // FIXED MUSIC MIXING LOGIC (From the working A-version)
    // ---------------------------------------------------------

    let mixedStream = videoStream;
    let musicCtx = null;
    let musicSrc = null;
    let musicGain = null;
    let musicDest = null;

    if (musicFile) {
      try {
        // decode audio file
        const audioBuffer = await decodeAudioFile(musicFile);

        // Create audio context for mixing
        musicCtx = new (window.AudioContext || window.webkitAudioContext)();
        musicDest = musicCtx.createMediaStreamDestination();

        // Create buffer source
        musicSrc = musicCtx.createBufferSource();
        musicSrc.buffer = audioBuffer;
        musicSrc.loop = true;

        // Volume control
        musicGain = musicCtx.createGain();
        musicGain.gain.value = Math.max(0, Math.min(1, musicVol));

        // connect
        musicSrc.connect(musicGain).connect(musicDest);

        // merge video + audio into one MediaStream
        mixedStream = new MediaStream([
          ...videoStream.getVideoTracks(),
          ...musicDest.stream.getAudioTracks(),
        ]);

      } catch (err) {
        console.warn("Music load/mix failed, continuing without music:", err);
      }
    }

    // Recorder
    const mimeCandidates = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm"
    ];
    let mime = "";
    for (const c of mimeCandidates) {
      if (MediaRecorder.isTypeSupported(c)) {
        mime = c;
        break;
      }
    }

    const chunks = [];
    const recorder = new MediaRecorder(
      mixedStream,
      mime ? { mimeType: mime, videoBitsPerSecond: 6000000 } : { videoBitsPerSecond: 6000000}
    );

    // collect data
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size) chunks.push(e.data);
    };

    // promise to wait for completion
    const done = new Promise((resolve) => (recorder.onstop = resolve));

    // Start recorder
    recorder.start(200);

    // Start music *after* recorder starts
    if (musicCtx && musicSrc) {
      try {
        if (musicCtx.state === "suspended") await musicCtx.resume();
        musicSrc.start(0);
      } catch (err) {
        console.warn("Music start failed:", err);
      }
    }


    const imageSources = await Promise.all(
      clipsToUse
        .filter((p) => p.type === "image")
        .map(
          (p) =>
            new Promise((res, rej) => {
              const img = new Image();
              img.onload = () => res({ meta: p, img });
              img.onerror = rej;
              img.src = p.url;
            })
        )
    );

    const videoSources = await Promise.all(
      clipsToUse
        .filter((p) => p.type === "video")
        .map(
          (p) =>
            new Promise((res) => {
              const v = document.createElement("video");
              v.src = p.url;
              v.preload = "auto";
              v.muted = true;
              v.playsInline = true;
              v.onloadedmetadata = () => res({ meta: p, video: v });
            })
        )
    );

    const sleepFrame = (frames) =>
      new Promise((r) => setTimeout(r, (1000 / FPS) * (frames || 1)));

    
    let currentProgress = 0;
    const totalFrames = Math.round(totalReelSecs * FPS);

    if (!startCaption) {
      for (let i = 0, n = Math.round(titleDur * FPS); i < n; i++) {
        drawTitleCard(ctx, canvas);
        await sleepFrame(1);
        currentProgress++;
        setGeneratingProgress(Math.round((currentProgress / totalFrames) * 100));
      }
    }

    if (startCaption) {
      for (let i = 0, n = Math.round(startCaptionDur * FPS); i < n; i++) {
        drawCaptionCard(ctx, canvas, startCaption);
        await sleepFrame(1);
        currentProgress++;
        setGeneratingProgress(Math.round((currentProgress / totalFrames) * 100));
      }
    }

    for (let clipIdx = 0; clipIdx < clipsToUse.length; clipIdx++) {
      const clip = clipsToUse[clipIdx];
      
      if (clip.type === "image") {
        const src = imageSources.find((s) => s.meta === clip);
        if (!src) continue;
        
        for (let i = 0, n = Math.round(clipDuration * FPS); i < n; i++) {
          const progress = i / n;
          drawImageFitVertical(
            ctx, 
            canvas, 
            src.img, 
            clip.rotation || 0, 
            clip.center,
            imageEffect,
            progress
          );
          await sleepFrame(1);
          currentProgress++;
          if (i % 5 === 0) {
            setGeneratingProgress(Math.round((currentProgress / totalFrames) * 100));
          }
        }
      } else if (clip.type === "video") {
        const src = videoSources.find((s) => s.meta === clip);
        if (!src) continue;

        const start = Math.max(0, clip.startTime || 0);
        const endTime = Math.min(
          (src.video.duration || start + clipDuration),
          start + clipDuration
        );

        src.video.currentTime = start;
        await waitEvent(src.video, "seeked");
        if (src.video.readyState < 2) {
          await Promise.race([waitEvent(src.video, "canplay"), waitEvent(src.video, "canplaythrough")]);
        }
        await src.video.play();

        let frameCount = 0;
        let cancel = () => {};
        await new Promise((resolve) => {
          const step = (_getNext, _ts, frame) => {
            const nowMedia = frame?.mediaTime ?? src.video.currentTime;
            drawImageFitVertical(
              ctx, 
              canvas, 
              src.video, 
              clip.rotation || 0, 
              clip.center,
              "none",
              0
            );
            currentProgress++;
            frameCount++;
            if (frameCount % 10 === 0) {
              setGeneratingProgress(Math.round((currentProgress / totalFrames) * 100));
            }

            if (nowMedia >= endTime - 0.01) {
              resolve();
              return;
            }
            cancel = onVideoFrame(src.video, step);
          };
          cancel = onVideoFrame(src.video, step);
        });

        cancel?.();
        src.video.pause();
      }
    }

    if (endCaption) {
      for (let i = 0, n = Math.round(endCaptionDur * FPS); i < n; i++) {
        drawCaptionCard(ctx, canvas, endCaption);
        await sleepFrame(1);
        currentProgress++;
        setGeneratingProgress(Math.round((currentProgress / totalFrames) * 100));
      }
    } else {
      for (let i = 0, n = Math.round(0.5 * FPS); i < n; i++) {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "white";
        ctx.font = "bold 60px system-ui, Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 20;
        ctx.fillText("Thanks for watching 🏃‍♀️", canvas.width / 2, canvas.height / 2);
        ctx.shadowColor = "transparent";
        await sleepFrame(1);
        currentProgress++;
        setGeneratingProgress(Math.round((currentProgress / totalFrames) * 100));
      }
    }

    recorder.stop();
    await done;

    try {
      musicSrc?.stop(0);
      if (musicCtx?.state !== "closed") await musicCtx.close();
    } catch {}

    const type = (chunks[0] && chunks[0].type) || "video/webm";
    const blob = new Blob(chunks, { type });
    const url = URL.createObjectURL(blob);
    setReelUrl(url);
    setIsGenerating(false);
    setGeneratingProgress(100);

    setTimeout(() => {
      previewVideoRef.current?.scrollIntoView({ behavior: "smooth" });
      setGeneratingProgress(0);
    }, 500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-4 bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
            <Camera className="w-6 h-6 text-purple-300" />
            <span className="text-white/90 font-medium">Team MotionLens</span>
          </div>
          <div className="flex items-center justify-center gap-4 mb-4">
            <Film className="w-14 h-14 text-purple-400" />
            <h1 className="text-5xl md:text-6xl font-bold text-white">RunReel</h1>
          </div>
          <p className="text-purple-200 text-lg">
            AI-Powered Running Detection • Vertical Reels • High Quality
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 mb-6 border border-white/20">
          <h2 className="text-white text-2xl font-semibold mb-3 flex items-center gap-2">
            <Upload className="w-6 h-6" /> Upload Your Media
          </h2>

          <div
            className="border-2 border-dashed border-purple-400/50 rounded-xl p-8 text-center cursor-pointer hover:border-purple-300 transition"
            onClick={() => fileInputRef.current?.click()}
          >
            <Camera className="w-16 h-16 text-purple-300 mx-auto mb-3" />
            <p className="text-white font-medium mb-1">
              Click to upload images & videos
            </p>
            <p className="text-purple-200 text-sm">Supported: JPG, PNG, MP4, MOV</p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,video/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>

          <div className="mt-5">
            <label className="text-white text-lg font-semibold mb-2 flex items-center gap-2">
              <Music2 className="w-5 h-5" />
              Optional: Add background music
            </label>
            <div
              className="border-2 border-dashed border-purple-400/50 rounded-xl p-4 text-center cursor-pointer hover:border-purple-300 transition"
              onClick={() => musicInputRef.current?.click()}
            >
              <p className="text-purple-200 text-sm">
                Supported: MP3, WAV, M4A, OGG (looped to fit reel)
              </p>
              <input
                ref={musicInputRef}
                type="file"
                accept="audio/*"
                onChange={handleMusicUpload}
                className="hidden"
              />
              {musicFile && <p className="text-green-200 mt-2">Selected: {musicFile.name}</p>}
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-white flex items-center gap-2 mb-1">
                  <Volume2 className="w-4 h-4" /> Music Volume
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={musicVol}
                  onChange={(e) => setMusicVol(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-purple-200 text-sm mt-1">{Math.round(musicVol * 100)}%</p>
              </div>
              <div>
                <label className="text-white mb-1 block">Fade In (sec)</label>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={fadeInSec}
                  onChange={(e) => setFadeInSec(Math.max(0, parseFloat(e.target.value || "0")))}
                  className="w-full rounded-md px-3 py-2"
                />
              </div>
              <div>
                <label className="text-white mb-1 block">Fade Out (sec)</label>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={fadeOutSec}
                  onChange={(e) => setFadeOutSec(Math.max(0, parseFloat(e.target.value || "0")))}
                  className="w-full rounded-md px-3 py-2"
                />
              </div>
            </div>
          </div>

          {files.length > 0 && (
            <div className="mt-6 flex flex-col md:flex-row gap-3">
              <button
                onClick={processFiles}
                disabled={isProcessing}
                className="flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-4 rounded-xl font-semibold hover:from-purple-700 hover:to-indigo-700 disabled:opacity-60 transition-all flex items-center justify-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" /> Analyzing...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" /> Detect Running Clips
                  </>
                )}
              </button>
              <button
                onClick={clearAll}
                className="px-6 py-4 rounded-xl font-semibold border-2 border-red-400 text-red-400 hover:bg-red-400/10 transition-colors flex items-center gap-2"
              >
                <Trash2 className="w-5 h-5" /> Clear
              </button>
            </div>
          )}

          {isProcessing && (
            <div className="mt-4">
              <p className="text-purple-200 text-sm mb-2">
                Analyzing poses... {Math.round(progress)}%
              </p>
              <div className="w-full h-3 bg-purple-900/50 rounded-full overflow-hidden">
                <div
                  style={{ width: `${progress}%` }}
                  className="h-3 bg-gradient-to-r from-purple-400 to-indigo-500 transition-all"
                />
              </div>
            </div>
          )}
        </div>

        {processedFiles.length > 0 && currentStep === "results" && (
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 mb-6 border border-white/20">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white text-2xl font-semibold flex items-center gap-2">
                <CheckCircle className="w-6 h-6 text-green-400" /> Running Clips (Chronological)
              </h3>
              <div className="bg-green-500/20 border border-green-400/40 px-3 py-2 rounded-lg text-green-200">
                {processedFiles.length} clips
              </div>
            </div>

            <p className="text-purple-200 text-sm mb-6">
              Avg confidence: {Math.round(stats.avgConfidence * 100)}% • Sorted by timestamp
            </p>

            <div className="bg-indigo-900/40 rounded-xl p-6 mb-6 border border-indigo-400/30">
              <h4 className="text-white text-lg font-semibold mb-4 flex items-center gap-2">
                <Type className="w-5 h-5" /> Custom Captions (Optional)
              </h4>
              
              <div className="space-y-4">
                <div>
                  <label className="text-white font-medium mb-2 block">
                    Start Caption
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., My Running Journey 2024"
                    value={startCaption}
                    onChange={(e) => setStartCaption(e.target.value)}
                    className="w-full rounded-lg px-4 py-3"
                  />
                  <p className="text-purple-200 text-sm mt-1">
                    Leave empty to use default title card
                  </p>
                </div>

                <div>
                  <label className="text-white font-medium mb-2 block">
                    End Caption
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., Keep Running! 💪"
                    value={endCaption}
                    onChange={(e) => setEndCaption(e.target.value)}
                    className="w-full rounded-lg px-4 py-3"
                  />
                  <p className="text-purple-200 text-sm mt-1">
                    Leave empty to use default end card
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-white font-medium mb-2 block">
                      Caption Duration Mode
                    </label>
                    <select
                      value={captionDuration}
                      onChange={(e) => setCaptionDuration(e.target.value)}
                      className="w-full rounded-lg px-4 py-3"
                    >
                      <option value="dynamic">Dynamic (based on text length)</option>
                      <option value="fixed">Fixed duration</option>
                    </select>
                  </div>

                  {captionDuration === "fixed" && (
                    <div>
                      <label className="text-white font-medium mb-2 block">
                        Fixed Duration (seconds)
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="15"
                        step="0.5"
                        value={fixedCaptionSecs}
                        onChange={(e) => setFixedCaptionSecs(Math.max(1, Math.min(15, parseFloat(e.target.value || "5"))))}
                        className="w-full rounded-lg px-4 py-3"
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-purple-900/30 rounded-xl p-6 mb-6 border border-purple-400/30">
              <h4 className="text-white text-lg font-semibold mb-4 flex items-center gap-2">
                <Film className="w-5 h-5" /> Reel Settings
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="text-white font-medium mb-2 block">
                    Number of clips (n)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max={processedFiles.length}
                    value={numClips}
                    onChange={(e) => setNumClips(Math.max(1, Math.min(processedFiles.length, parseInt(e.target.value || "1"))))}
                    className="w-full rounded-lg px-4 py-3 text-lg font-semibold"
                  />
                  <p className="text-purple-200 text-sm mt-2">
                    Use first {Math.min(numClips, processedFiles.length)} clips
                  </p>
                </div>
                <div>
                  <label className="text-white font-medium mb-2 block">
                    Duration per clip (t) - seconds
                  </label>
                  <input
                    type="number"
                    min="0.5"
                    max="10"
                    step="0.5"
                    value={timePerClip}
                    onChange={(e) => setTimePerClip(Math.max(0.5, Math.min(10, parseFloat(e.target.value || "1"))))}
                    className="w-full rounded-lg px-4 py-3 text-lg font-semibold"
                  />
                  <p className="text-purple-200 text-sm mt-2">
                    Each clip will be {timePerClip}s
                  </p>
                </div>
                <div>
                  <label className="text-white font-medium mb-2 block">
                    Image Effect
                  </label>
                  <select
                    value={imageEffect}
                    onChange={(e) => setImageEffect(e.target.value)}
                    className="w-full rounded-lg px-4 py-3 text-lg font-semibold"
                  >
                    <option value="zoomIn">Zoom In (Subtle)</option>
                    <option value="zoomOut">Zoom Out (Subtle)</option>
                    <option value="none">None</option>
                  </select>
                  <p className="text-purple-200 text-sm mt-2">
                    Effect for static images
                  </p>
                </div>
              </div>
              <div className="mt-4 bg-indigo-900/40 rounded-lg p-4 border border-indigo-400/30">
                <p className="text-white font-semibold">
                  📱 Vertical Format (720x1280) • 30 FPS • High Quality (6 Mbps)
                </p>
                <p className="text-purple-200 text-sm mt-1">
                  Total Duration: ~{Math.round(
                    (startCaption ? getCaptionDuration(startCaption) : 0.75) +
                    (Math.min(numClips, processedFiles.length) * timePerClip) +
                    (endCaption ? getCaptionDuration(endCaption) : 0.5)
                  )}s • Person-Centered Cropping • No Distortion
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
              {processedFiles.map((p, idx) => (
                <div
                  key={idx}
                  className="relative rounded-lg overflow-hidden border border-purple-400/20 bg-purple-900/30"
                >
                  <img
                    src={p.displayUrl || p.thumbUrl || p.url}
                    alt={`clip-${idx}`}
                    className="w-full h-48 object-cover"
                    style={{
                      transform: `rotate(${p.rotation || 0}deg)`,
                    }}
                  />
                  <div className="absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full font-semibold">
                    {Math.round(p.confidence * 100)}%
                  </div>
                  <button
                    onClick={() => rotateClip(idx)}
                    className="absolute bottom-2 right-2 bg-blue-500 hover:bg-blue-600 text-white p-2 rounded-full transition-colors shadow-lg"
                    title="Rotate 90°"
                  >
                    <RotateCw className="w-4 h-4" />
                  </button>
                  <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-semibold">
                    #{idx + 1}
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={generateReel}
              disabled={isGenerating}
              className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white px-6 py-4 rounded-xl font-bold hover:from-green-700 hover:to-emerald-700 disabled:opacity-60 transition-all flex items-center justify-center gap-3 shadow-lg"
            >
              {isGenerating ? (
                <>
                  <Loader className="w-6 h-6 animate-spin" /> Generating High-Quality Reel...
                </>
              ) : (
                <>
                  <Film className="w-6 h-6" /> Generate Vertical Reel (HD)
                </>
              )}
            </button>

            {isGenerating && (
              <div className="mt-4">
                <p className="text-purple-200 text-sm mb-2">
                  🎬 Rendering high-quality vertical reel... {Math.round(generatingProgress)}%
                </p>
                <div className="w-full h-4 bg-purple-900/50 rounded-full overflow-hidden">
                  <div
                    style={{ width: `${generatingProgress}%` }}
                    className="h-4 bg-gradient-to-r from-green-400 to-emerald-500 transition-all duration-300"
                  />
                </div>
                <p className="text-purple-300 text-xs mt-2">
                  This may take a few moments for high quality output...
                </p>
              </div>
            )}
          </div>
        )}

        <canvas
          ref={renderCanvasRef}
          width={REEL_WIDTH}
          height={REEL_HEIGHT}
          style={{ display: "none" }}
        />

        {reelUrl && (
          <div
            ref={previewVideoRef}
            className="bg-white/10 backdrop-blur-md rounded-2xl p-6 mt-6 border border-white/20"
          >
            <h3 className="text-white text-xl font-semibold mb-3 flex items-center gap-2">
              <CheckCircle className="w-6 h-6 text-green-400" /> 
              Your Vertical Reel is Ready!
            </h3>
            <div className="flex justify-center mb-4">
              <video 
                src={reelUrl} 
                controls 
                className="rounded-xl border-2 border-purple-400/30 shadow-2xl"
                style={{ maxHeight: "70vh", width: "auto" }}
              />
            </div>
            <div className="bg-green-900/30 rounded-lg p-4 mb-4 border border-green-400/30">
              <p className="text-green-200 text-sm">
                ✨ <strong>Quality:</strong> 720x1280 (Full HD) • 30 FPS • 6 Mbps<br/>
                📱 <strong>Format:</strong> Vertical (9:16) • Perfect for Instagram Reels, TikTok, YouTube Shorts<br/>
                🎯 <strong>Features:</strong> Smart cropping centered on runner • No distortion • High bitrate
              </p>
            </div>
            <div className="flex gap-3 justify-center flex-wrap">
              <a
                href={reelUrl}
                download="runreel-vertical-hd.webm"
                className="px-6 py-3 rounded-lg bg-green-600 text-white font-semibold hover:bg-green-700 transition-colors shadow-lg flex items-center gap-2"
              >
                <Film className="w-5 h-5" />
                Download HD Vertical Reel
              </a>
              <button
                onClick={() => setReelUrl(null)}
                className="px-6 py-3 rounded-lg border-2 border-red-400 text-red-400 hover:bg-red-400/10 transition-colors flex items-center gap-2"
              >
                <Trash2 className="w-5 h-5" />
                Clear & Create New
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RunReelApp;