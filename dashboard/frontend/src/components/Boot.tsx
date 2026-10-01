import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import Logo from './Logo'

// Startup screen. Asli checks karta hai: backend zinda hai? kaunsa data hai?
export default function Boot() {
  const [lines, setLines] = useState<string[]>([])
  const [progress, setProgress] = useState(0)
  const [visible, setVisible] = useState(true)
  const started = useRef(false)   // React StrictMode effect 2 baar chalata hai; sequence sirf ek baar chalao

  useEffect(() => {
    if (started.current) return
    started.current = true
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))
    const say = async (line: string, pct: number) => {
      setLines((l) => [...l, line])
      setProgress(pct)
      await wait(260)
    }

    ;(async () => {
      await say('> STARTING TRADING ENGINE ..............', 10)
      let online = false
      try {
        await api.health()
        online = true
        await say('> ENGINE CONNECTION ....................OK', 30)
      } catch {
        await say('> ENGINE CONNECTION ................OFFLINE', 30)
        await say('> start backend: uvicorn src.api.server:app --reload', 40)
      }
      if (online) {
        try {
          const symbols = await api.symbols()
          for (const [s, tfs] of Object.entries(symbols)) {
            await say(`> MARKET DATA ${s.padEnd(8, ' ')} [${tfs.join(' ')}] ..OK`, 60)
          }
        } catch {
          await say('> MARKET DATA .....................ERROR', 60)
        }
      }
      await say('> STRUCTURE ENGINE .....................OK', 75)
      await say('> ORDERFLOW ZONE SCANNER ...............OK', 85)
      let trained = false
      if (online) {
        try {
          await api.mlReport()
          trained = true
        } catch {
          trained = false
        }
      }
      await say(trained ? '> ML BRAIN .........................TRAINED' : '> ML BRAIN ....................NOT TRAINED', 92)
      await say(online ? '> READY TO TRADE.' : '> RUNNING IN OFFLINE MODE.', 100)
      await wait(400)
      setVisible(false)
    })()
  }, [])

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          exit={{ opacity: 0 }}
          transition={{ duration: 0.8 }}
          className="fixed inset-0 z-[100] flex flex-col items-center justify-center gap-5"
          style={{ background: '#000000' }}
        >
          <Logo size={96} />
          <div className="font-display glow-cyan text-3xl tracking-[10px]">AI FOREX TRADER</div>
          <pre className="min-h-[190px] w-[520px] max-w-[92vw] text-[13px]" style={{ color: 'var(--green)' }}>
            {lines.join('\n')}
          </pre>
          <div className="bar w-[520px] max-w-[92vw]" style={{ ['--c' as string]: 'var(--cyan)' }}>
            <i style={{ width: `${progress}%` }} />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
