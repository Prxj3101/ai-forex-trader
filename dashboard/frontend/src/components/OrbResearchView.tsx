import OrbBrainPanel from './OrbBrainPanel'
import OrbHistoryChart from './OrbHistoryChart'
import OrbResearchPanel from './OrbResearchPanel'
import OrbStatsPage from './OrbStatsPage'

/** Research -> ORB: ORB strategy ka sab kuch ek jagah (zone bot se alag). */
export default function OrbResearchView({ connected }: { connected: boolean }) {
  return (
    <>
      <OrbStatsPage connected={connected} />
      <OrbHistoryChart connected={connected} />
      <OrbResearchPanel connected={connected} />
      <OrbBrainPanel connected={connected} />
    </>
  )
}
